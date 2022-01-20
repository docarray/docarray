from typing import (
    Sequence,
    Iterable,
)

from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    def _get_doc_by_offset(self, index: int) -> 'Document':
        ...

    def _get_doc_by_id(self, id: str) -> 'Document':
        ...

    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        ...

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._get_docs_by_offsets(range(len(self))[_slice])
