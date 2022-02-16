from typing import (
    Union,
    Optional,
    TYPE_CHECKING,
    List,
)
from .... import Document, DocumentArray

if TYPE_CHECKING:
    import numpy as np


class FindMixin:
    def find(
        self,
        query: Union['DocumentArray', 'Document', 'np.ndarray'],
        limit: Optional[Union[int, float]] = 20,
        only_id: bool = False,
        **kwargs,
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns approximate nearest neighbors given an input query.

        :param query: the query documents to search.
        :param limit: the number of results to get for each query document in search.
        :param only_id: if set, then returning matches will only contain ``id``
        :param kwargs: other kwargs.

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise
            a list of DocumentArrays containing the closest Document objects for each of the queries in `query`.
        """
        from ....math import ndarray

        if limit is not None:
            if limit <= 0:
                raise ValueError(f'`limit` must be larger than 0, receiving {limit}')
            else:
                limit = int(limit)

        if isinstance(query, Document):
            query = ndarray.to_numpy_array(query.embedding)
        elif isinstance(query, DocumentArray):
            query = ndarray.to_numpy_array(query.embeddings)

        n_rows, _ = ndarray.get_array_rows(query)
        if n_rows == 1:
            query = query.reshape(1, -1)

        _, match_docs = self._pqlite._search_documents(
            query, limit=limit, include_metadata=not only_id, **kwargs
        )
        if n_rows == 1:
            return match_docs[0]
        return match_docs
