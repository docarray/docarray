from typing import TYPE_CHECKING, TypeVar, List, Union, Optional, Dict, Sequence

if TYPE_CHECKING:
    import numpy as np
    import tensorflow
    import torch

    # Define the expected input type that your ANN search supports
    MilvusArrayType = TypeVar(
        'MilvusArrayType',
        np.ndarray,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )
    from docarray import Document, DocumentArray


class FindMixin:
    def _find(
        self,
        query: 'MilvusArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        param=None,
        **kwargs
    ) -> List['DocumentArray']:
        """Returns `limit` approximate nearest neighbors given a batch of input queries.
        If the query is a single query, should return a DocumentArray, otherwise a list of DocumentArrays containing
        the closest Documents for each query.
        """
        if param is None:
            param = dict()
        kwargs = self._update_kwargs_from_config('consistency_level', **kwargs)
        with self.loaded_collection():
            results = self._collection.search(
                data=query,
                anns_field='embedding',
                limit=limit,
                expr=filter,
                param=param,
                output_fields=['serialized'],
                **kwargs,
            )
        return self._docs_from_search_response(results, distance=self._config.distance)

    def _filter(self, filter, limit=10, **kwargs):
        kwargs = self._update_kwargs_from_config('consistency_level', **kwargs)
        with self.loaded_collection():
            results = self._collection.query(
                expr=filter,
                limit=limit,
                output_fields=['serialized'],
                **kwargs,
            )
        return self._docs_from_query_response(results)[:limit]
