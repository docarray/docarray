from typing import (
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    List,
    Union,
    Optional,
    Dict,
)

from redis.commands.search.query import NumericFilter, Query
from redis.commands.search.field import VectorField

import numpy as np

from .... import Document, DocumentArray
from ....math import ndarray
from ....math.ndarray import to_numpy_array
from ....score import NamedScore
from ....array.mixins.find import FindMixin as BaseFindMixin


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
            query_params.update(filter)
        results = (
            self._client.ft(index_name=self.index_name).search(q, query_params).docs
        )

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob)
            doc.embedding = np.frombuffer(res.embedding, dtype=np.float32)
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
