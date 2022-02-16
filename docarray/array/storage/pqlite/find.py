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
        only_id: bool = False,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given an input query.

        :param query: the query documents to search.
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """
        if limit is not None:
            if limit <= 0:
                raise ValueError(f'`limit` must be larger than 0, receiving {limit}')
            else:
                limit = int(limit)
        self._pqlite.search(query, limit=limit, include_metadata=not only_id, **kwargs)
        return [q.matches for q in query]
