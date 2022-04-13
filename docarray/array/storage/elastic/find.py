import functools
from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
    Union,
    Dict,
    Callable,
    Optional,
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
    from ....typing import T, ArrayType

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
        self, query: str, limit: int = 10, index: str = 'text'
    ):
        """
        Return key-word matches for the input query

        :param query: text used for key-word search
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """

        resp = self._client.search(
            index=self._config.index_name,
            query={'match': {index: query}},
            source=['id', 'blob', 'text'],
        )
        list_of_hits = resp['hits']['hits']

        da = DocumentArray()
        for result in list_of_hits[:limit]:
            doc = Document.from_base64(result['_source']['blob'])
            doc.scores['score'] = NamedScore(value=result['_score'])
            da.append(doc)

        return da

    def _find(
        self,
        query: Union['ElasticArrayType', str, List[str]],
        limit: int = 10,
        index: str = 'text',
        **kwargs,
    ) -> List['DocumentArray']:
        """Returns approximate nearest neighbors given a batch of input queries if the input is an 'ElasticArrayType'.
           Returns exact key-word search if the input is `str` or `List[str]` if the input.

        :param query: input supported to be stored in Elastic. This includes any from the list '[np.ndarray, tensorflow.Tensor, torch.Tensor, Sequence[float]]'
        :param limit: number of retrieved items

        :return: DocumentArray containing the closest documents to the query if it is a single query, otherwise a list of DocumentArrays containing
           the closest Document objects for each of the queries in `query`.
        """
        if isinstance(query, str):
            search_method = functools.partial(
                self._find_similar_documents_from_text, index=index
            )
            num_rows = 1
            query = [query]
        elif isinstance(query, list) and isinstance(query[0], str):
            search_method = functools.partial(
                self._find_similar_documents_from_text, index=index
            )
            num_rows = len(query)
        else:
            search_method = self._find_similar_vectors
            query = np.array(query)
            num_rows, n_dim = ndarray.get_array_rows(query)
            if n_dim != 2:
                query = query.reshape((num_rows, -1))

        if num_rows == 1:
            # if it is a list do query[0]
            return [search_method(query[0], limit=limit)]
        else:
            closest_docs = []
            for q in query:
                da = search_method(q, limit=limit)
                closest_docs.append(da)
            return closest_docs
