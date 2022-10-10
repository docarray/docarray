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
            expr=None,
            param=param,
            output_fields=['serialized'],
            **kwargs
        )
        self._collection.release()
        return self._docs_from_search_response(results)
