import itertools
from typing import (
    Sequence,
    Iterable,
    Any,
)

from ..base.getsetdel import BaseGetSetDelMixin
from ..memory.backend import needs_id2offset_rebuild
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    @needs_id2offset_rebuild
    def _del_docs_by_mask(self, mask: Sequence[bool]):
        self._data = list(itertools.compress(self._data, (not _i for _i in mask)))

    @needs_id2offset_rebuild
    def _del_docs_by_slice(self, _slice: slice):
        del self._data[_slice]

    def _del_doc_by_id(self, _id: str):
        self._del_doc_by_offset(self._id2offset[_id])

    @needs_id2offset_rebuild
    def _del_doc_by_offset(self, offset: int):
        del self._data[offset]

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        old_id = self._data[offset].id
        self._id2offset[value.id] = offset
        self._data[offset] = value
        self._id2offset.pop(old_id)

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        old_idx = self._id2offset.pop(_id)
        self._data[old_idx] = value
        self._id2offset[value.id] = old_idx

    @needs_id2offset_rebuild
    def _set_docs_by_slice(self, _slice: slice, value: Sequence['Document']):
        self._data[_slice] = value

    def _set_doc_attr_by_offset(self, offset: int, attr: str, value: Any):
        if attr == 'id' and value is None:
            raise ValueError(
                'setting the ID of a Document stored in a DocumentArray to None is not allowed'
            )

        setattr(self._data[offset], attr, value)

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        return self._data[offset]

    def _get_doc_by_id(self, _id: str) -> 'Document':
        return self._data[self._id2offset[_id]]

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._data[_slice]

    def _clear_storage(self):
        self._data.clear()
        self._id2offset.clear()

    def _load_offset2ids(self):
        ...

    def _save_offset2ids(self):
        ...

    _set_doc = _set_doc_by_id
    _del_doc = _del_doc_by_id
    _del_all_docs = _clear_storage
