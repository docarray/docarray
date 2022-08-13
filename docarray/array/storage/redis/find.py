from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
from docarray import Document, DocumentArray
from docarray.array.mixins.find import FindMixin as BaseFindMixin
from docarray.math import ndarray
from docarray.math.ndarray import to_numpy_array
from docarray.score import NamedScore

from redis.commands.search.query import NumericFilter, Query

if TYPE_CHECKING:
    import tensorflow
    import torch

    RedisArrayType = TypeVar(
        'RedisArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
        Dict,
    )


class FindMixin(BaseFindMixin):
    def _find_similar_vectors(
        self, query: 'RedisArrayType', filter: Optional[Dict] = None, limit=10
    ):
        q = (
            Query("*=>[KNN " + str(limit) + " @embedding $vec AS vector_score]")
            .sort_by('vector_score')
            .dialect(2)
        )

        query_params = {"vec": to_numpy_array(query).astype(np.float32).tobytes()}
        if filter:
            f = self._build_fiter(filter)
            q.add_filter(f)
        results = self._client.ft().search(q, query_params).docs

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            doc.scores['score'] = NamedScore(value=res.vector_score)
            da.append(doc)
        return da

    def _find(
        self,
        query: 'RedisArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        **kwargs
    ) -> List['DocumentArray']:

        query = np.array(query)
        num_rows, n_dim = ndarray.get_array_rows(query)
        if n_dim != 2:
            query = query.reshape((num_rows, -1))

        return [
            self._find_similar_vectors(q, filter=filter, limit=limit) for q in query
        ]

    def _find_with_filter(self, filter: Dict, limit: Optional[Union[int, float]] = 20):

        if filter:
            s = self._build_query_str(filter)
            q = Query(s)
            q.paging(0, limit)

        results = self._client.ft().search(q).docs

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            da.append(doc)
        return da

    def _filter(
        self, filter: Dict, limit: Optional[Union[int, float]] = 20
    ) -> 'DocumentArray':

        return self._find_with_filter(filter, limit=limit)

    # TODO return NumericFilter or List[NumericFilter]
    def _build_fiter(self, filter: Dict) -> NumericFilter:

        INF = "+inf"
        NEG_INF = "-inf"

        if filter['operator'] == 'gt':
            f = NumericFilter(filter['key'], filter['value'], INF, minExclusive=True)
        elif filter['operator'] == 'gte':
            f = NumericFilter(filter['key'], filter['value'], INF)
        elif filter['operator'] == 'lt':
            f = NumericFilter(
                filter['key'], NEG_INF, filter['value'], maxExclusive=True
            )
        elif filter['operator'] == 'lte':
            f = NumericFilter(filter['key'], NEG_INF, filter['value'])

        return f

    def _build_query_str(self, filter: Dict) -> str:
        INF = "+inf"
        NEG_INF = "-inf"

        if filter['operator'] == 'gt':
            s = "@{}:[({} {}]".format(filter['key'], filter['value'], INF)
        elif filter['operator'] == 'gte':
            s = "@{}:[{} {}]".format(filter['key'], filter['value'], INF)
        elif filter['operator'] == 'lt':
            s = "@{}:[{} ({}]".format(filter['key'], NEG_INF, filter['value'])
        elif filter['operator'] == 'lte':
            s = "@{}:[{} {}]".format(filter['key'], NEG_INF, filter['value'])

        return s
