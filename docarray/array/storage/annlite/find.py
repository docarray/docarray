from typing import (
    Union,
    Optional,
    TYPE_CHECKING,
    List,
    Dict,
)

if TYPE_CHECKING:
    import numpy as np

from docarray import DocumentArray


class FindMixin:
    def _find(
        self,
        query: 'np.ndarray',
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given an input query.

        :param query: the query documents to search.
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param filter: filter query used for pre-filtering
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """
        from docarray.math import ndarray

        n_rows, _ = ndarray.get_array_rows(query)
        if n_rows == 1:
            query = query.reshape(1, -1)

        _, match_docs = self._annlite._search_documents(
            query, limit=limit, filter=filter or {}, include_metadata=not only_id
        )

        return match_docs

    def _filter(
        self,
        filter: Dict,
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
    ) -> 'DocumentArray':
        """Returns a subset of documents by filtering by the given filter (`Annlite` filter).

        :param filter: the input filter to apply in each stored document
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :return: a `DocumentArray` containing the `Document` objects that verify the filter.
        """

        docs = self._annlite.filter(
            filter=filter,
            limit=limit,
            include_metadata=not only_id,
        )
        return DocumentArray(docs)
