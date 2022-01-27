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

    def _del_doc_by_id(self, _id: str):
        offset = self._offset2ids.get_offset_by_id(_id)
        self._offset2ids.del_at_offset(offset, commit=True)
        self._pqlite.delete([_id])

    def _del_doc_by_offset(self, offset: int):
        _id = self._offset2ids.get_id_by_offset(offset)
        self._pqlite.delete([_id])

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        self._offset2ids.set_at_offset(offset, value.id)
        docs = DocumentArrayInMemory([value])
        if docs.embeddings is None:
            docs.embeddings = np.zeros((1, self._pqlite.dim))
        self._pqlite.update(docs)

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        docs = DocumentArrayInMemory([value])
        if docs.embeddings is None:
            docs.embeddings = np.zeros((1, self._pqlite.dim))
        self._pqlite.update(docs)

    def _get_doc_by_offset(self, index: int) -> 'Document':
        doc_id = self._offset2ids.get_id_by_offset(index)
        if doc_id is not None:
            return self._pqlite.get_doc_by_id(doc_id)

    def _get_doc_by_id(self, id: str) -> 'Document':
        return self._pqlite.get_doc_by_id(id)

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        return [self._get_doc_by_offset(offset) for offset in offsets]

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._get_docs_by_offsets(range(len(self))[_slice])
