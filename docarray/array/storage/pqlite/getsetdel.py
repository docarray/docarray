from typing import (
    Sequence,
    Iterable,
)

import numpy as np

from ..base.getsetdel import BaseGetSetDelMixin
from ...memory import DocumentArrayInMemory
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    # essential methods start

    def _get_doc_by_id(self, _id: str) -> 'Document':
        doc = self._pqlite.get_doc_by_id(_id)
        if doc is None:
            raise KeyError(f'Can not find Document with id=`{_id}`')
        return doc

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        self._to_numpy_embedding(value)
        docs = DocumentArrayInMemory([value])
        self._pqlite.update(docs)

    def _del_doc_by_id(self, _id: str):
        self._pqlite.delete([_id])
