from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
    Union,
    Optional,
    Dict,
)

import numpy as np

from .... import Document, DocumentArray
from ....math import ndarray
from ....math.helper import EPSILON
from ....math.ndarray import to_numpy_array
from ....score import NamedScore
from ....array.mixins.find import FindMixin as BaseFindMixin

if TYPE_CHECKING:
    import tensorflow
    import torch

    ElasticArrayType = TypeVar(
        'ElasticArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )


class FindMixin(BaseFindMixin):
    def _find_similar_vectors(self, query: 'ElasticArrayType', limit=10):
        query = to_numpy_array(query)
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        resp = self._client.knn_search(
            index=self._config.index_name,
            knn={
                'field': 'embedding',
                'query_vector': query,
                'k': limit,
                'num_candidates': 10000,
            },
        )
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            doc.embedding = result['_source']['embedding']
            da.append(doc)

        return da

    def _find_similar_documents_from_text(
        self, query: str, index: str = 'text', limit: int = 10
    ):
        """
        Return keyword matches for the input query

        :param query: text used for keyword search
        :param limit: number of items to be retrieved

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """

        resp = self._client.search(
            index=self._config.index_name,
            query={'match': {index: query}},
            source=['id', 'blob', 'text'],
            size=limit,
        )
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits[:limit]:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            da.append(doc)

        return da

    def _find_by_text(
        self, query: Union[str, List[str]], index: str = 'text', limit: int = 10
    ):
        if isinstance(query, str):
            query = [query]

        return [
            self._find_similar_documents_from_text(q, index=index, limit=limit)
            for q in query
        ]

    def _find(
        self,
        query: 'ElasticArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.

        :param query: input supported to be stored in Elastic. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items
        :param filter: filter query used for pre-filtering

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """
        if filter is not None:
            raise ValueError(
                'Filtered vector search is not supported for ElasticSearch backend'
            )
        query = np.array(query)
        num_rows, n_dim = ndarray.get_array_rows(query)
        if n_dim != 2:
            query = query.reshape((num_rows, -1))

        return [self._find_similar_vectors(q, limit=limit) for q in query]
