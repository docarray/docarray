import itertools
from typing import (
    Sequence,
    Iterable,
    Any,
)

from pandas import Series

from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    def _del_docs_by_mask(self, mask: Sequence[bool]):
        self._data.drop(itertools.compress(self._data.index, mask), inplace=True)

    def _del_all_docs(self):
        self._data = Series()

    def _del_docs_by_slice(self, _slice: slice):
        self._data.drop(self._data.index[_slice], inplace=True)

    def _del_doc_by_id(self, _id: str):
        self._data.drop(_id, inplace=True)

    def _del_doc_by_offset(self, offset: int):
        self._data.drop(self._data.index[offset], inplace=True)

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        # TODO: what if value.id is different from self._data.index[offset]?
        self._data[offset] = value

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        # TODO: what if value.id is different from _id?
        self._data[_id] = value

    def _set_docs_by_slice(self, _slice: slice, value: Sequence['Document']):
        # TODO: what if value.id is different from _id?
        if not isinstance(value, Sequence):
            raise TypeError('can only assign an iterable')
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
        # TODO: what if value.id is different from _id?
        setattr(self._data[_id], attr, value)

    def _get_doc_by_offset(self, offset: int) -> 'Document':
        return self._data[offset]

    def _get_doc_by_id(self, _id: str) -> 'Document':
        return self._data[_id]

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._data[_slice]

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        if isinstance(offsets, tuple):
            return self._data[list(offsets)]
        return self._data[offsets]

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        return self._data[ids]
