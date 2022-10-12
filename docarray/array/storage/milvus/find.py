from typing import TYPE_CHECKING, TypeVar, List, Union, Optional, Dict

if TYPE_CHECKING:
    import numpy as np

    # Define the expected input type that your ANN search supports
    MilvusArrayType = TypeVar(
        'MilvusArrayType', np.ndarray, list
    )  # TODO(johannes) test torch, tf, etc.


class FindMixin:
    def _find(
        self,
        query: 'MilvusArrayType',
        limit: int = 10,
        filter: Optional[Dict] = None,
        param=None,
        **kwargs
    ) -> Union['DocumentArray', List['DocumentArray']]:
        """Returns `limit` approximate nearest neighbors given a batch of input queries.
        If the query is a single query, should return a DocumentArray, otherwise a list of DocumentArrays containing
        the closest Documents for each query.
        """
        if param is None:
            param = dict()
        self._collection.load()
        results = self._collection.search(
            data=query,
            anns_field='embedding',
            limit=limit,
            expr=filter,
            param=param,
            output_fields=['serialized'],
            **kwargs
        )
        self._collection.release()
        return self._docs_from_search_response(results)

    def _filter(self, filter, limit=10, **kwargs):
        # TODO(johannes) apply this consistency level handling everywhere; spin it out into a helper function
        kwargs_consistency_level = kwargs.get('consistency_level', None)
        consistency_level = (
            kwargs_consistency_level
            if kwargs_consistency_level
            else self._config.consistency_level
        )
        self._collection.load()
        results = self._collection.query(
            expr=filter,
            limit=limit,
            output_fields=['serialized'],
            consistency_level=consistency_level,
            **kwargs
        )
        self._collection.release()
        return self._docs_from_query_respone(results)[:limit]
