import warnings
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

if TYPE_CHECKING:  # pragma: no cover
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
        filter: Optional[Union[str, Dict]] = None,
        limit: Union[int, float] = 20,
        **kwargs,
    ):

        if filter:
            query_str = _get_redis_filter_query(filter)
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
            doc.scores['score'] = NamedScore(value=float(res.vector_score))
            da.append(doc)
        return da

    def _find(
        self,
        query: 'RedisArrayType',
        limit: Union[int, float] = 20,
        filter: Optional[Union[str, Dict]] = None,
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

    def _find_with_filter(
        self,
        filter: Union[str, Dict],
        limit: Union[int, float] = 20,
    ):
        query_str = _get_redis_filter_query(filter)
        q = Query(query_str)
        q.paging(0, limit)

        results = self._client.ft(index_name=self._config.index_name).search(q).docs

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            da.append(doc)
        return da

    def _filter(
        self,
        filter: Union[str, Dict],
        limit: Union[int, float] = 20,
    ) -> 'DocumentArray':

        return self._find_with_filter(filter, limit=limit)

    def _find_by_text(
        self,
        query: Union[str, List[str]],
        index: str = 'text',
        filter: Optional[Union[str, Dict]] = None,
        limit: Union[int, float] = 20,
        **kwargs,
    ):
        if isinstance(query, str):
            query = [query]

        return [
            self._find_similar_documents_from_text(
                q,
                index=index,
                filter=filter,
                limit=limit,
                **kwargs,
            )
            for q in query
        ]

    def _find_similar_documents_from_text(
        self,
        query: str,
        index: str = 'text',
        filter: Optional[Union[str, Dict]] = None,
        limit: Union[int, float] = 20,
        **kwargs,
    ):
        query_str = _build_query_str(query)

        if filter:
            filter_str = _get_redis_filter_query(filter)
        else:
            filter_str = ''

        scorer = kwargs.get('scorer', 'BM25')
        if scorer not in [
            'BM25',
            'TFIDF',
            'TFIDF.DOCNORM',
            'DISMAX',
            'DOCSCORE',
            'HAMMING',
        ]:
            raise ValueError(
                f'Expecting a valid text similarity ranking algorithm, got {scorer} instead'
            )

        q = Query(f'@{index}:{query_str} {filter_str}').scorer(scorer).paging(0, limit)

        results = self._client.ft(index_name=self._config.index_name).search(q).docs

        da = DocumentArray()
        for res in results:
            doc = Document.from_base64(res.blob.encode())
            da.append(doc)
        return da


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


def _build_query_str(query):
    query_str = '|'.join(query.split(' '))
    return query_str


def _get_redis_filter_query(filter: Union[str, Dict]):
    if isinstance(filter, dict):
        warnings.warn(
            "Dict syntax for redis filter will be deprecated, use string literals instead",
            DeprecationWarning,
        )
        nodes = _build_query_nodes(filter)
        query_str = intersect(*nodes).to_string()
    elif isinstance(filter, str):
        query_str = filter
    else:
        raise ValueError(f'Unexpected type of filter: {type(filter)}, expected str')

    return query_str
