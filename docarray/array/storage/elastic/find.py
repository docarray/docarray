from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
)

import numpy as np

from .... import Document, DocumentArray
from ....math import ndarray
from ....math.helper import EPSILON
from ....math.ndarray import to_numpy_array
from ....score import NamedScore

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


class FindMixin:
    def _find_similar_vectors(self, query: 'ElasticArrayType', limit=10):
        query = to_numpy_array(query)
        is_all_zero = np.all(query == 0)
        if is_all_zero:
            query = query + EPSILON

        resp = self._client.knn_search(
            index=self._config.index_name,
            knn={
                "field": "embedding",
                "query_vector": query,
                "k": limit,
                "num_candidates": 10000,
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

    def _find(
        self, query: 'ElasticArrayType', limit: int = 10, **kwargs
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries.
        :param query: input supported to be stored in Elastic. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.

        """

        num_rows, _ = ndarray.get_array_rows(query)

        if num_rows == 1:
            return [self._find_similar_vectors(query[0], limit=limit)]
        else:
            closest_docs = []
            for q in query:
                da = self._find_similar_vectors(q, limit=limit)
                closest_docs.append(da)
            return closest_docs
