from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
from docarray import Document, DocumentArray
from docarray.array.mixins.find import FindMixin as BaseFindMixin
from docarray.math import ndarray
from docarray.math.ndarray import to_numpy_array
from docarray.score import NamedScore

from redis.commands.search.query import Query
from redis.commands.search.querystring import (
    DistjunctUnion,
    IntersectNode,
    equal,
    ge,
    gt,
    intersect,
    le,
    lt,
    union,
)

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
        limit: int = 20,
        **kwargs,
    ):

        if filter:
            nodes = _build_query_nodes(filter)
            query_str = intersect(*nodes).to_string()
        else:
            query_str = '*'

        q = (
            Query(f'({query_str})=>[KNN {limit} @embedding $vec AS vector_score]')
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
        limit: int = 20,
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

    def _find_with_filter(self, filter: Dict, limit: int = 20):
        nodes = _build_query_nodes(filter)
        query_str = intersect(*nodes).to_string()
        q = Query(query_str)
        q.paging(0, limit)

        results = self._client.ft(index_name=self._config.index_name).search(q).docs

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            da.append(doc)
        return da

    def _filter(self, filter: Dict, limit: int = 20) -> 'DocumentArray':

        return self._find_with_filter(filter, limit=limit)


def _build_query_node(key, condition):
    operator = list(condition.keys())[0]
    value = condition[operator]

    query_dict = {}

    if operator in ['$ne', '$eq']:
        if isinstance(value, bool):
            query_dict[key] = equal(int(value))
        elif isinstance(value, (int, float)):
            query_dict[key] = equal(value)
        else:
            query_dict[key] = value
    elif operator == '$gt':
        query_dict[key] = gt(value)
    elif operator == '$gte':
        query_dict[key] = ge(value)
    elif operator == '$lt':
        query_dict[key] = lt(value)
    elif operator == '$lte':
        query_dict[key] = le(value)
    else:
        raise ValueError(
            f'Expecting filter operator one of $gt, $gte, $lt, $lte, $eq, $ne, $and OR $or, got {operator} instead'
        )

    if operator == '$ne':
        return DistjunctUnion(**query_dict)
    return IntersectNode(**query_dict)


def _build_query_nodes(filter):
    nodes = []
    for k, v in filter.items():
        if k == '$and':
            children = _build_query_nodes(v)
            node = intersect(*children)
            nodes.append(node)
        elif k == '$or':
            children = _build_query_nodes(v)
            node = union(*children)
            nodes.append(node)
        else:
            child = _build_query_node(k, v)
            nodes.append(child)

    return nodes
