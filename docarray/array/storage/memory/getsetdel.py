from typing import (
    Sequence,
    Iterable,
    Any,
)

from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    def _del_doc_by_id(self, _id: str):
        del self._data[_id]

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if _id != value.id:
            del self._data[_id]
        self._data[value.id] = value

    def _set_doc_value_pairs(
        self, docs: Iterable['Document'], values: Sequence['Document']
    ):
        docs = list(docs)

        for _d, _v in zip(docs, values):
            _d._data = _v._data

    def _get_doc_by_id(self, _id: str) -> 'Document':
        return self._data[_id]

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable['Document']:
        return (self._data[_id] for _id in ids)

    def _clear_storage(self):
        self._data.clear()

    def _load_offset2ids(self):
        self._offset2ids = Offset2ID()

    def _save_offset2ids(self):
        ...
