from typing import (
    Union,
    Optional,
    TYPE_CHECKING,
    List,
)

if TYPE_CHECKING:
    from .... import DocumentArray


class FindMixin:
    def search(
        self,
        query: 'DocumentArray',
        limit: Optional[Union[int, float]] = 20,
        **kwargs,
    ) -> 'DocumentArray':
        self._pqlite.search(query, limit=limit)
        return query
