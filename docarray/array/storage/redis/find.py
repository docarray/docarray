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
        self,
        query: 'RedisArrayType',
        filter: Optional[Dict] = None,
        limit: Optional[Union[int, float]] = 20,
        **kwargs,
    ):

        query_str = self._build_query_str(filter) if filter else "*"

        q = (
            Query(f'{query_str}=>[KNN {limit} @embedding $vec AS vector_score]')
            .sort_by('vector_score')
            .paging(0, limit)
            .dialect(2)
        )

        query_params = {'vec': to_numpy_array(query).astype(np.float32).tobytes()}
        results = (
            self._client.ft(index_name=self._config.index_name)
            .search(q, query_params)
            .docs
        )

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            doc.scores['score'] = NamedScore(value=res.vector_score)
            da.append(doc)
        return da

    def _find(
        self,
        query: 'RedisArrayType',
        limit: Optional[Union[int, float]] = 20,
        filter: Optional[Dict] = None,
        **kwargs,
    ) -> List['DocumentArray']:

        query = np.array(query)
        num_rows, n_dim = ndarray.get_array_rows(query)
        if n_dim != 2:
            query = query.reshape((num_rows, -1))

        return [
            self._find_similar_vectors(q, filter=filter, limit=limit, **kwargs)
            for q in query
        ]

    def _find_with_filter(self, filter: Dict, limit: Optional[Union[int, float]] = 20):
        s = self._build_query_str(filter)
        q = Query(s)
        q.paging(0, limit)

        results = self._client.ft(index_name=self._config.index_name).search(q).docs

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            da.append(doc)
        return da

    def _filter(
        self, filter: Dict, limit: Optional[Union[int, float]] = 20
    ) -> 'DocumentArray':

        return self._find_with_filter(filter, limit=limit)

    def _build_query_str(self, filter: Dict) -> str:
        INF = "+inf"
        NEG_INF = "-inf"
        s = "("

        for key in filter:
            operator = list(filter[key].keys())[0]
            value = filter[key][operator]
            if operator == '$gt':
                s += f"@{key}:[({value} {INF}] "
            elif operator == '$gte':
                s += f"@{key}:[{value} {INF}] "
            elif operator == '$lt':
                s += f"@{key}:[{NEG_INF} ({value}] "
            elif operator == '$lte':
                s += f"@{key}:[{NEG_INF} {value}] "
            elif operator == '$eq':
                if type(value) is int:
                    s += f"@{key}:[{value} {value}] "
                elif type(value) is bool:
                    s += f"@{key}:[{int(value)} {int(value)}] "
                else:
                    s += f"@{key}:{value} "
            elif operator == '$ne':
                if type(value) is int:
                    s += f"-@{key}:[{value} {value}] "
                elif type(value) is bool:
                    s += f"-@{key}:[{int(value)} {int(value)}] "
                else:
                    s += f"-@{key}:{value} "
        s += ")"

        return s
