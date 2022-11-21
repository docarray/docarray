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

from docarray import Document, DocumentArray
from docarray.math import ndarray
from docarray.math.helper import EPSILON
from docarray.math.ndarray import to_numpy_array
from docarray.score import NamedScore
from docarray.array.mixins.find import FindMixin as BaseFindMixin


if TYPE_CHECKING:  # pragma: no cover
    import tensorflow
    import torch

    OpenSearchArrayType = TypeVar(
        'OpenSearchArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
        Dict,
    )


class FindMixin(BaseFindMixin):
    def _find_similar_vectors(
        self,
        query: 'OpenSearchArrayType',
        filter: Optional[Dict] = None,
        limit=10,
        **kwargs,
    ):
        """
        Return vector search results for the input query. `script_score` will be used in filter_field is set.
        :param query: query vector used for vector search
        :param filter: filter query used for post-filtering
        :param limit: number of items to be retrieved
        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """
        query = to_numpy_array(query)
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        filter_query = {'match_all': {}}

        if filter:
            filter_query = {'bool': {'filter': filter}}

        knn_query = {
            'size': limit,
            'query': {
                'script_score': {
                    'query': filter_query,
                    'script': {
                        'lang': 'knn',
                        'source': 'knn_score',
                        'params': {
                            'field': 'embedding',
                            'query_value': query,
                            'space_type': self._get_distance_metric(
                                kwargs.get('distance')
                            ),
                        },
                    },
                }
            },
        }

        resp = self._client.search(index=self._config.index_name, body=knn_query)
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            doc.embedding = result['_source']['embedding']
            da.append(doc)

        return da

    def _get_distance_metric(self, distance=None):
        return distance if distance else self._config.distance

    def _find_similar_documents_from_text(
        self,
        query: str,
        index: str = 'text',
        filter: Union[dict, list] = None,
        limit: int = 10,
    ):
        """
        Return keyword matches for the input query
        :param query: text used for keyword search
        :param limit: number of items to be retrieved
        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """

        query = {
            '_source': ['id', 'blob', 'text'],
            'size': limit,
            'query': {
                "bool": {
                    "must": [
                        {"match": {index: query}},
                    ],
                    'filter': filter,
                }
            },
        }

        resp = self._client.search(index=self._config.index_name, body=query)
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits[:limit]:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            da.append(doc)

        return da

    def _find_by_text(
        self,
        query: Union[str, List[str]],
        index: str = 'text',
        filter: Union[dict, list] = None,
        limit: int = 10,
    ):
        if isinstance(query, str):
            query = [query]

        return [
            self._find_similar_documents_from_text(
                q,
                index=index,
                filter=filter,
                limit=limit,
            )
            for q in query
        ]

    def _find(
        self,
        query: 'OpenSearchArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be stored in OpenSearch. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items
        :param filter: filter query used for pre-filtering
        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """
        query = np.array(query).astype(np.float)
        num_rows, n_dim = ndarray.get_array_rows(query)
        if n_dim != 2:
            query = query.reshape((num_rows, -1))

        return [
            self._find_similar_vectors(q, filter=filter, limit=limit, **kwargs)
            for q in query
        ]

    def _find_with_filter(self, query: Dict, limit: Optional[Union[int, float]] = 20):
        resp = self._client.search(
            index=self._config.index_name, body={'query': query, 'size': limit}
        )
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits[:limit]:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            da.append(doc)

        return da

    def _filter(
        self, query: Dict, limit: Optional[Union[int, float]] = 20
    ) -> 'DocumentArray':

        return self._find_with_filter(query, limit=limit)
