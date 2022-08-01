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
    def _get_index(self, subindex_name):
        is_root_index = subindex_name is None or subindex_name == '@r'
        if is_root_index:
            return self._annlite
        if subindex_name in self._subindices.keys():
            return self._subindices[subindex_name]._annlite
        raise ValueError(
            f"No subindex available for on='{subindex_name}'. "
            f'To create a subindex, pass `subindex_configs` when creating the DocumentArray.'
        )

    def _find(
        self,
        query: 'np.ndarray',
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
        filter: Optional[Dict] = None,
        on: Optional[str] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given an input query.

        :param query: the query documents to search.
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param filter: filter query used for pre-filtering
        :param on: specifies a subindex to search on. If set, then the returned DocumentArray will be retrieved from the given subindex.
        :param kwargs: other kwargs.

        :return: a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """
        from docarray.math import ndarray

        _annlite = self._get_index(on)

        n_rows, _ = ndarray.get_array_rows(query)
        if n_rows == 1:
            query = query.reshape(1, -1)

        _, match_docs = _annlite._search_documents(
            query, limit=limit, filter=filter or {}, include_metadata=not only_id
        )

        return match_docs

    def _filter(
        self,
        filter: Dict,
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
        on: Optional[str] = None,
    ) -> 'DocumentArray':
        """Returns a subset of documents by filtering by the given filter (`Annlite` filter).

        :param filter: the input filter to apply in each stored document
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param on: specifies a subindex to search on. If set, then the returned DocumentArray will be retrieved from the given subindex.
        :return: a `DocumentArray` containing the `Document` objects that verify the filter.
        """

        _annlite = self._get_index(on)

        docs = _annlite.filter(
            filter=filter,
            limit=limit,
            include_metadata=not only_id,
        )
        return DocumentArray(docs)
