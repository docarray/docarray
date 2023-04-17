# mypy: ignore-errors
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from elastic_transport import NodeConfig
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
from pydantic import parse_obj_as

import docarray.typing
from docarray import BaseDoc
from docarray.index.abstract import BaseDocIndex, _ColumnInfo, _raise_not_composable
from docarray.typing import AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available
from docarray.utils.find import _FindResult, _FindResultBatched

TSchema = TypeVar('TSchema', bound=BaseDoc)
T = TypeVar('T', bound='ElasticDocIndex')

ELASTIC_PY_VEC_TYPES: List[Any] = [list, tuple, np.ndarray, AbstractTensor]

if is_torch_available():
    import torch

    ELASTIC_PY_VEC_TYPES.append(torch.Tensor)

if is_tf_available():
    import tensorflow as tf  # type: ignore

    from docarray.typing import TensorFlowTensor

    ELASTIC_PY_VEC_TYPES.append(tf.Tensor)
    ELASTIC_PY_VEC_TYPES.append(TensorFlowTensor)


class ElasticDocIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        """Initialize ElasticDocIndex"""
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(ElasticDocIndex.DBConfig, self._db_config)

        # ElasticSearch client creation
        if self._db_config.index_name is None:
            id = uuid.uuid4().hex
            self._db_config.index_name = 'index__' + id

        self._index_name = self._db_config.index_name

        self._client = Elasticsearch(
            hosts=self._db_config.hosts,
            **self._db_config.es_config,
        )

        # ElasticSearh index setup
        self._index_vector_params = ('dims', 'similarity', 'index')
        self._index_vector_options = ('m', 'ef_construction')

        mappings: Dict[str, Any] = {
            'dynamic': True,
            '_source': {'enabled': 'true'},
            'properties': {},
        }
        mappings.update(self._db_config.index_mappings)

        for col_name, col in self._column_infos.items():
            if col.db_type == 'dense_vector' and (
                not col.n_dim and col.config['dims'] < 0
            ):
                self._logger.info(
                    f'Not indexing column {col_name}, the dimensionality is not specified'
                )
                continue

            mappings['properties'][col_name] = self._create_index_mapping(col)

        # print(mappings['properties'])
        if self._client.indices.exists(index=self._index_name):
            self._client_put_mapping(mappings)
        else:
            self._client_create(mappings)

        if len(self._db_config.index_settings):
            self._client_put_settings(self._db_config.index_settings)

        self._refresh(self._index_name)

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################
    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, outer_instance, **kwargs):
            super().__init__()
            self._outer_instance = outer_instance
            self._query: Dict[str, Any] = {
                'query': defaultdict(lambda: defaultdict(list))
            }

        def build(self, *args, **kwargs) -> Any:
            """Build the elastic search query object."""
            if len(self._query['query']) == 0:
                del self._query['query']
            elif 'knn' in self._query:
                self._query['knn']['filter'] = self._query['query']
                del self._query['query']

            return self._query

        def find(
            self,
            query: Union[AnyTensor, BaseDoc],
            search_field: str = 'embedding',
            limit: int = 10,
            num_candidates: Optional[int] = None,
        ):
            """
            Find k-nearest neighbors of the query.

            :param query: query vector for KNN/ANN search. Has single axis.
            :param search_field: name of the field to search on
            :param limit: maximum number of documents to return per query
            :param num_candidates: number of candidates
            :return: self
            """
            self._outer_instance._validate_search_field(search_field)
            if isinstance(query, BaseDoc):
                query_vec = BaseDocIndex._get_values_by_column([query], search_field)[0]
            else:
                query_vec = query
            query_vec_np = BaseDocIndex._to_numpy(self._outer_instance, query_vec)
            self._query['knn'] = self._outer_instance._form_search_body(
                query_vec_np,
                limit,
                search_field,
                num_candidates,
            )['knn']

            return self

        # filter accepts Leaf/Compound query clauses
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
        def filter(self, query: Dict[str, Any], limit: int = 10):
            """Find documents in the index based on a filter query

            :param query: the query to execute
            :param limit: maximum number of documents to return
            :return: self
            """
            self._query['size'] = limit
            self._query['query']['bool']['filter'].append(query)
            return self

        def text_search(self, query: str, search_field: str = 'text', limit: int = 10):
            """Find documents in the index based on a text search query

            :param query: The text to search for
            :param search_field: name of the field to search on
            :param limit: maximum number of documents to find
            :return: self
            """
            self._outer_instance._validate_search_field(search_field)
            self._query['size'] = limit
            self._query['query']['bool']['must'].append(
                {'match': {search_field: query}}
            )
            return self

        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('filter_batched')
        text_search_batched = _raise_not_composable('text_search_batched')

    def build_query(self, **kwargs) -> QueryBuilder:
        """
        Build a query for ElasticDocIndex.
        :param kwargs: parameters to forward to QueryBuilder initialization
        :return: QueryBuilder object
        """
        return self.QueryBuilder(self, **kwargs)

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of ElasticDocIndex."""

        hosts: Union[
            str, List[Union[str, Mapping[str, Union[str, int]], NodeConfig]], None
        ] = 'http://localhost:9200'
        index_name: Optional[str] = None
        es_config: Dict[str, Any] = field(default_factory=dict)
        index_settings: Dict[str, Any] = field(default_factory=dict)
        index_mappings: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of ElasticDocIndex."""

        default_column_config: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
        chunk_size: int = 500

        def __post_init__(self):
            self.default_column_config = {
                'binary': {},
                'boolean': {},
                'keyword': {},
                'long': {},
                'integer': {},
                'short': {},
                'byte': {},
                'double': {},
                'float': {},
                'half_float': {},
                'scaled_float': {},
                'unsigned_long': {},
                'dates': {},
                'alias': {},
                'object': {},
                'flattened': {},
                'nested': {},
                'join': {},
                'integer_range': {},
                'float_range': {},
                'long_range': {},
                'double_range': {},
                'date_range': {},
                'ip_range': {},
                'ip': {},
                'version': {},
                'histogram': {},
                'text': {},
                'annotated_text': {},
                'completion': {},
                'search_as_you_type': {},
                'token_count': {},
                'sparse_vector': {},
                'rank_feature': {},
                'rank_features': {},
                'geo_point': {},
                'geo_shape': {},
                'point': {},
                'shape': {},
                'percolator': {},
                # `None` is not a Type, but we allow it here anyway
                None: {},  # type: ignore
            }
            self.default_column_config['dense_vector'] = self.dense_vector_config()

        def dense_vector_config(self):
            """Get the dense vector config."""
            config = {
                'dims': -1,
                'index': True,
                'similarity': 'cosine',  # 'l2_norm', 'dot_product', 'cosine'
                'm': 16,
                'ef_construction': 100,
                'num_candidates': 10000,
            }

            return config

    ###############################################
    # Implementation of abstract methods          #
    ###############################################

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """
        for allowed_type in ELASTIC_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return 'dense_vector'

        elastic_py_types = {
            docarray.typing.ID: 'keyword',
            docarray.typing.AnyUrl: 'keyword',
            bool: 'boolean',
            int: 'integer',
            float: 'float',
            str: 'text',
            bytes: 'binary',
            dict: 'object',
        }

        for type in elastic_py_types.keys():
            if issubclass(python_type, type):
                return elastic_py_types[type]

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _index(
        self,
        column_to_data: Mapping[str, Generator[Any, None, None]],
        refresh: bool = True,
        chunk_size: Optional[int] = None,
    ):
        data = self._transpose_col_value_dict(column_to_data)
        requests = []

        for row in data:
            request = {
                '_index': self._index_name,
                '_id': row['id'],
            }
            for col_name, col in self._column_infos.items():
                if col.db_type == 'dense_vector' and np.all(row[col_name] == 0):
                    row[col_name] = row[col_name] + 1.0e-9
                if row[col_name] is None:
                    continue
                request[col_name] = row[col_name]
            requests.append(request)

        _, warning_info = self._send_requests(requests, chunk_size)
        for info in warning_info:
            warnings.warn(str(info))

        if refresh:
            self._refresh(self._index_name)

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        return self._client.count(index=self._index_name)['count']

    def _del_items(
        self,
        doc_ids: Sequence[str],
        chunk_size: Optional[int] = None,
    ):
        requests = []
        for _id in doc_ids:
            requests.append(
                {'_op_type': 'delete', '_index': self._index_name, '_id': _id}
            )

        _, warning_info = self._send_requests(requests, chunk_size)

        # raise warning if some ids are not found
        if warning_info:
            ids = [info['delete']['_id'] for info in warning_info]
            warnings.warn(f'No document with id {ids} found')

        self._refresh(self._index_name)

    def _get_items(self, doc_ids: Sequence[str]) -> Sequence[TSchema]:
        accumulated_docs = []
        accumulated_docs_id_not_found = []

        es_rows = self._client_mget(doc_ids)['docs']

        for row in es_rows:
            if row['found']:
                doc_dict = row['_source']
                accumulated_docs.append(doc_dict)
            else:
                accumulated_docs_id_not_found.append(row['_id'])

        # raise warning if some ids are not found
        if accumulated_docs_id_not_found:
            warnings.warn(f'No document with id {accumulated_docs_id_not_found} found')

        return accumulated_docs

    def execute_query(self, query: Dict[str, Any], *args, **kwargs) -> Any:
        """
        Execute a query on the ElasticDocIndex.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        resp = self._client.search(index=self._index_name, **query)
        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=parse_obj_as(NdArray, scores))

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        body = self._form_search_body(query, limit, search_field)

        resp = self._client_search(**body)

        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=parse_obj_as(NdArray, scores))

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        request = []
        for query in queries:
            head = {'index': self._index_name}
            body = self._form_search_body(query, limit, search_field)
            request.extend([head, body])

        responses = self._client_msearch(request)

        das, scores = zip(
            *[self._format_response(resp) for resp in responses['responses']]
        )
        return _FindResultBatched(documents=list(das), scores=scores)

    def _filter(
        self,
        filter_query: Dict[str, Any],
        limit: int,
    ) -> List[Dict]:
        resp = self._client_search(query=filter_query, size=limit)

        docs, _ = self._format_response(resp)

        return docs

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> List[List[Dict]]:
        request = []
        for query in filter_queries:
            head = {'index': self._index_name}
            body = {'query': query, 'size': limit}
            request.extend([head, body])

        responses = self._client_msearch(request)
        das, _ = zip(*[self._format_response(resp) for resp in responses['responses']])

        return list(das)

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        body = self._form_text_search_body(query, limit, search_field)
        resp = self._client_search(**body)

        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        request = []
        for query in queries:
            head = {'index': self._index_name}
            body = self._form_text_search_body(query, limit, search_field)
            request.extend([head, body])

        responses = self._client_msearch(request)
        das, scores = zip(
            *[self._format_response(resp) for resp in responses['responses']]
        )
        return _FindResultBatched(documents=list(das), scores=scores)

    ###############################################
    # Helpers                                     #
    ###############################################

    def _create_index_mapping(self, col: '_ColumnInfo') -> Dict[str, Any]:
        """Create a new HNSW index for a column, and initialize it."""

        index = {'type': col.config['type'] if 'type' in col.config else col.db_type}

        if col.db_type == 'dense_vector':
            for k in self._index_vector_params:
                index[k] = col.config[k]
            if col.n_dim:
                index['dims'] = col.n_dim
            index['index_options'] = dict(
                (k, col.config[k]) for k in self._index_vector_options
            )
            index['index_options']['type'] = 'hnsw'
        return index

    def _send_requests(
        self,
        request: Iterable[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[List[Dict], List[Any]]:
        """Send bulk request to Elastic and gather the successful info"""

        accumulated_info = []
        warning_info = []
        for success, info in parallel_bulk(
            self._client,
            request,
            raise_on_error=False,
            raise_on_exception=False,
            chunk_size=chunk_size if chunk_size else self._runtime_config.chunk_size,  # type: ignore
            **kwargs,
        ):
            if not success:
                warning_info.append(info)
            else:
                accumulated_info.append(info)

        return accumulated_info, warning_info

    def _form_search_body(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
        num_candidates: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not num_candidates:
            num_candidates = self._runtime_config.default_column_config['dense_vector'][
                'num_candidates'
            ]
        body = {
            'size': limit,
            'knn': {
                'field': search_field,
                'query_vector': query,
                'k': limit,
                'num_candidates': num_candidates,
            },
        }
        return body

    def _form_text_search_body(
        self, query: str, limit: int, search_field: str = ''
    ) -> Dict[str, Any]:
        body = {
            'size': limit,
            'query': {
                'bool': {
                    'must': {'match': {search_field: query}},
                }
            },
        }
        return body

    def _format_response(self, response: Any) -> Tuple[List[Dict], List[Any]]:
        docs = []
        scores = []
        for result in response['hits']['hits']:
            if not isinstance(result, dict):
                result = result.to_dict()

            if result.get('_source', None):
                doc_dict = result['_source']
            else:
                doc_dict = result['fields']
            doc_dict['id'] = result['_id']
            docs.append(doc_dict)
            scores.append(result['_score'])

        return docs, [parse_obj_as(NdArray, np.array(s)) for s in scores]

    def _refresh(self, index_name: str):
        self._client.indices.refresh(index=index_name)

    ###############################################
    # API Wrappers                                #
    ###############################################

    def _client_put_mapping(self, mappings: Dict[str, Any]):
        self._client.indices.put_mapping(
            index=self._index_name, properties=mappings['properties']
        )

    def _client_create(self, mappings: Dict[str, Any]):
        self._client.indices.create(index=self._index_name, mappings=mappings)

    def _client_put_settings(self, settings: Dict[str, Any]):
        self._client.indices.put_settings(index=self._index_name, settings=settings)

    def _client_mget(self, ids: Sequence[str]):
        return self._client.mget(index=self._index_name, ids=ids)

    def _client_search(self, **kwargs):
        return self._client.search(index=self._index_name, **kwargs)

    def _client_msearch(self, request: List[Dict[str, Any]]):
        return self._client.msearch(index=self._index_name, searches=request)
