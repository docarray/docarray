from typing import (
    Sequence,
    Iterable,
)
import numpy as np
from ...memory import DocumentArrayInMemory
from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    # essential methods start

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        offset = len(self) + offset if offset < 0 else offset
        doc_id = self._offset2ids.get_id_by_offset(offset)
        doc = self._pqlite.get_doc_by_id(doc_id) if doc_id else None
        if doc is None:
            raise IndexError('index out of range')
        return doc

    def _get_doc_by_id(self, _id: str) -> 'Document':
        doc = self._pqlite.get_doc_by_id(_id)
        if doc is None:
            raise KeyError(f'Can not find Document with id=`{_id}`')
        return doc

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        ids = self._offset2ids.get_ids_by_offsets(offsets)
        return self._get_docs_by_ids(ids)

    def _get_docs_by_ids(self, ids: str) -> Iterable['Document']:
        for _id in ids:
            yield self._get_doc_by_id(_id)

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._get_docs_by_offsets(range(len(self))[_slice])

    def _get_docs_by_mask(self, mask: Sequence[bool]):
        offsets = [i for i, m in enumerate(mask) if m is True]
        return self._get_docs_by_offsets(offsets)

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        offset = len(self) + offset if offset < 0 else offset
        self._offset2ids.set_at_offset(offset, value.id)
        docs = DocumentArrayInMemory([value])
        if docs.embeddings is None:
            docs.embeddings = np.zeros((1, self._pqlite.dim))
        self._pqlite.update(docs)

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        offset = self._offset2ids.get_offset_by_id(_id)
        self._set_doc_by_offset(offset, value)

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Iterable['Document']
    ):
        for _d, _v in zip(docs, values):
            self._set_doc_by_id(_d.id, _v)

    def _del_doc_by_id(self, _id: str):
        offset = self._offset2ids.get_offset_by_id(_id)
        self._offset2ids.del_at_offset(offset, commit=True)
        self._pqlite.delete([_id])

    def _del_doc_by_offset(self, offset: int):
        offset = len(self) + offset if offset < 0 else offset
        _id = self._offset2ids.get_id_by_offset(offset)
        self._offset2ids.del_at_offset(offset)
        self._pqlite.delete([_id])

    def _del_doc_by_offsets(self, offsets: Sequence[int]):
        ids = []
        for offset in offsets:
            ids.append(self._offset2ids.get_id_by_offset(offset))

        self._offset2ids.del_at_offsets(offsets)
        self._pqlite.delete(ids)

    def _del_docs_by_slice(self, _slice: slice):
        offsets = range(len(self))[_slice]
        self._del_doc_by_offsets(offsets)

    def _del_docs_by_mask(self, mask: Sequence[bool]):
        offsets = [i for i, m in enumerate(mask) if m is True]
        self._del_doc_by_offsets(offsets)
