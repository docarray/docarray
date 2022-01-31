import itertools
from typing import (
    Sequence,
    Iterable,
    Any,
)
from ..memory.backend import needs_id2offset_rebuild
from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document, DocumentArray


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    @needs_id2offset_rebuild
    def _del_docs_by_mask(self, mask: Sequence[bool]):
        self._data = list(itertools.compress(self._data, (not _i for _i in mask)))

    def _del_all_docs(self):
        self._data.clear()
        self._id2offset.clear()

    @needs_id2offset_rebuild
    def _del_docs_by_slice(self, _slice: slice):
        del self._data[_slice]

    def _del_doc_by_id(self, _id: str):
        self._del_doc_by_offset(self._id2offset[_id])

    @needs_id2offset_rebuild
    def _del_doc_by_offset(self, offset: int):
        del self._data[offset]

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        self._data[offset] = value
        self._id2offset[value.id] = offset

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        old_idx = self._id2offset.pop(_id)
        self._data[old_idx] = value
        self._id2offset[value.id] = old_idx

    @needs_id2offset_rebuild
    def _set_docs_by_slice(self, _slice: slice, value: Sequence['Document']):
        self._data[_slice] = value

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Sequence['Document']
    ):
        docs = list(docs)

        for _d, _v in zip(docs, values):
            _d._data = _v._data

    def _set_doc_attr_by_offset(self, offset: int, attr: str, value: Any):
        setattr(self._data[offset], attr, value)

    def _set_doc_attr_by_id(self, _id: str, attr: str, value: Any):
        if attr == 'id':
            old_idx = self._id2offset.pop(_id)
            setattr(self._data[old_idx], attr, value)
            self._id2offset[value] = old_idx
        else:
            setattr(self._data[self._id2offset[_id]], attr, value)

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        return self._data[offset]

    def _get_doc_by_id(self, _id: str) -> 'Document':
        return self._data[self._id2offset[_id]]

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._data[_slice]

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        return (self._data[t] for t in offsets)

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        return (self._data[self._id2offset[t]] for t in ids)
