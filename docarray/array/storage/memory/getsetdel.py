import itertools
from typing import (
    Sequence,
    Iterable,
    Any,
)

import pandas as pd
from pandas import Series

from .helper import _get_docs_ids
from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document


def _insert_at_series(s: Series, index, value) -> Series:
    s1 = s[:index]
    s2 = s[index:]
    s1[value.id] = value
    return s1.append(s2)


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
        if offset < 0:
            offset = offset + len(self._data)
        old_id = self._data.index[offset]
        if value.id != old_id:
            self._data.drop(old_id, inplace=True)
            self._data = _insert_at_series(self._data, offset, value)
        else:
            self._data[offset] = value

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if value.id != _id:
            offset = self._data.index.get_loc(_id)
            self._data.drop(_id, inplace=True)
            self._data = _insert_at_series(self._data, offset, value)
        else:
            self._data[_id] = value

    def _set_docs_by_slice(self, _slice: slice, value: Sequence['Document']):
        if not isinstance(value, Sequence):
            raise TypeError('can only assign an iterable')
        _docs, ids = _get_docs_ids(value)
        start, step, end = (
            _slice.start or 0,
            _slice.step or 1,
            _slice.stop or len(self._data),
        )
        self._data = (
            self._data[:start].append(Series(_docs, index=ids)).append(self._data[end:])
        )

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Sequence['Document']
    ):
        docs = list(docs)

        for _d, _v in zip(docs, values):
            _d._data = _v._data

    def _set_doc_attr_by_offset(self, offset: int, attr: str, value: Any):
        setattr(self._data[offset], attr, value)

    def _set_doc_attr_by_id(self, _id: str, attr: str, value: Any):
        doc = self._data[_id]
        setattr(doc, attr, value)
        if attr == 'id' and _id != value:
            offset = self._data.index.get_loc(_id)
            self._data.drop(_id, inplace=True)
            self._data = _insert_at_series(self._data, offset, doc)

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
        if isinstance(ids, tuple):
            return self._data[list(ids)]
        return self._data[ids]
