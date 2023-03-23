import os
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
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk

import docarray.typing
from docarray import BaseDocument
from docarray.index.abstract import (
    BaseDocumentIndex,
    _ColumnInfo,
    _FindResultBatched,
    _raise_not_composable,
)
from docarray.typing import AnyTensor
from docarray.utils.find import _FindResult
from docarray.utils.misc import torch_imported

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='ElasticDocumentIndex')

ELASTIC_PY_VEC_TYPES: List[Any] = [np.ndarray]

if torch_imported:
    import torch

    ELASTIC_PY_VEC_TYPES.append(torch.Tensor)


class ElasticDocumentIndex(BaseDocumentIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(ElasticDocumentIndex.DBConfig, self._db_config)

        if self._db_config.index_name is None:
            id = uuid.uuid4().hex
            self._db_config.index_name = 'index__' + id

        self._index_name = self._db_config.index_name

        self._client = Elasticsearch(
            hosts=self._db_config.hosts,
            **self._db_config.es_config,
        )

        # compatibility
        self._server_version = self._client.info()['version']['number']
        if int(self._server_version.split('.')[0]) >= 8:
            os.environ['ELASTIC_CLIENT_APIVERSIONING'] = '1'

        # ElasticSearh index setup
        self._index_vector_params = ('dims',)

        body: Dict[str, Any] = {
            'mappings': {
                'dynamic': True,
                '_source': {'enabled': 'true'},
                'properties': {},
            }
        }

        for col_name, col in self._column_infos.items():
            body['mappings']['properties'][col_name] = self._create_index_mapping(col)

        if self._client.indices.exists(index=self._index_name):
            self._client.indices.put_mapping(
                index=self._index_name, body=body['mappings']
            )
        else:
            self._client.indices.create(index=self._index_name, body=body)

        if len(self._db_config.index_settings):
            self._client.indices.put_settings(
                index=self._index_name, body=self._db_config.index_settings
            )

        self._refresh(self._index_name)

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################

    class QueryBuilder(BaseDocumentIndex.QueryBuilder):
        def __init__(self, outer_instance, **kwargs):
            super().__init__()
            self._outer_instance = outer_instance
            self._query: Dict[str, Any] = {
                'query': defaultdict(lambda: defaultdict(list))
            }

        def build(self, *args, **kwargs) -> Any:
            if (
                'script_score' in self._query['query']
                and 'bool' in self._query['query']
                and len(self._query['query']['bool']) > 0
            ):
                self._query['query']['script_score']['query'] = {}
                self._query['query']['script_score']['query']['bool'] = self._query[
                    'query'
                ]['bool']
                del self._query['query']['bool']

            return self._query

        def find(
            self,
            query: Union[AnyTensor, BaseDocument],
            search_field: str = 'embedding',
            limit: int = 10,
        ):
            if isinstance(query, BaseDocument):
                query_vec = BaseDocumentIndex._get_values_by_column(
                    [query], search_field
                )[0]
            else:
                query_vec = query
            query_vec_np = BaseDocumentIndex._to_numpy(self._outer_instance, query_vec)
            self._query['size'] = limit
            self._query['query']['script_score'] = {
                'query': {'match_all': {}},
                'script': {
                    'source': f'cosineSimilarity(params.query_vector, \'{search_field}\') + 1.0',
                    'params': {'query_vector': query_vec_np},
                },
            }
            return self

        def filter(self, query: Dict[str, Any], limit: int = 10):
            self._query['size'] = limit
            self._query['query']['bool']['filter'].append(query)
            return self

        def text_search(self, query: str, search_field: str = 'text', limit: int = 10):
            self._query['size'] = limit
            self._query['query']['bool']['must'].append(
                {'match': {search_field: query}}
            )
            return self

        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('find_batched')
        text_search_batched = _raise_not_composable('text_search')

    def build_query(self, **kwargs) -> QueryBuilder:
        """
        Build a query for this DocumentIndex.
        """
        return self.QueryBuilder(self, **kwargs)

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):
        hosts: Union[str, List[str], None] = 'http://localhost:9200'
        index_name: Optional[str] = None
        es_config: Dict[str, Any] = field(default_factory=dict)
        index_settings: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
        default_column_config: Dict[Any, Dict[str, Any]] = field(
            default_factory=lambda: {
                'dense_vector': {'dims': 128},
                'keyword': {},
                'boolean': {},
                'integer': {},
                'float': {},
                'text': {},
                # `None` is not a Type, but we allow it here anyway
                None: {},  # type: ignore
            }
        )
        chunk_size: int = 500

    ###############################################
    # Implementation of abstract methods          #
    ###############################################

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""

        for allowed_type in ELASTIC_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return 'dense_vector'

        elastic_py_types = {
            bool: 'boolean',
            int: 'integer',
            float: 'float',
            str: 'text',
            docarray.typing.ID: 'keyword',
        }

        if python_type in elastic_py_types:
            return elastic_py_types[python_type]

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _index(
        self,
        column_to_data: Dict[str, Generator[Any, None, None]],
        refresh: bool = True,
        chunk_size: Optional[int] = None,
    ):

        data = self._transpose_col_value_dict(column_to_data)  # type: ignore
        requests = []

        for row in data:
            request = {
                '_index': self._index_name,
                '_id': row['id'],
            }
            for col_name, col in self._column_infos.items():
                if col.db_type == np.ndarray and np.all(row[col_name] == 0):
                    row[col_name] = row[col_name] + 1.0e-9
                request[col_name] = row[col_name]
            requests.append(request)

        _, warning_info = self._send_requests(requests, chunk_size)
        for info in warning_info:
            warnings.warn(str(info))

        if refresh:
            self._refresh(self._index_name)

    def num_docs(self) -> int:
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

        es_rows = self._client.mget(
            index=self._index_name,
            body={'ids': doc_ids},
        )['docs']

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
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        resp = self._client.search(index=self._index_name, body=query)
        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        if int(self._server_version.split('.')[0]) >= 8:
            warnings.warn(
                'You are using Elasticsearch 8.0+ and the current client is 7.10.1. HNSW based vector search is not supported and the find method has a default implementation.'
            )

        body = {
            'size': limit,
            'query': {
                'script_score': {
                    'query': {'match_all': {}},
                    'script': {
                        'source': f'cosineSimilarity(params.query_vector, \'{search_field}\') + 1.0',
                        'params': {'query_vector': query},
                    },
                }
            },
        }

        resp = self._client.search(
            index=self._index_name,
            body=body,
        )

        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        result_das = []
        result_scores = []

        for query in queries:
            documents, scores = self._find(query, limit, search_field)
            result_das.append(documents)
            result_scores.append(scores)

        return _FindResultBatched(documents=result_das, scores=np.array(result_scores))  # type: ignore

    def _filter(
        self,
        filter_query: Dict[str, Any],
        limit: int,
    ) -> List[Dict]:
        body = {
            'size': limit,
            'query': filter_query,
        }

        resp = self._client.search(
            index=self._index_name,
            body=body,
        )

        docs, _ = self._format_response(resp)

        return docs

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> List[List[Dict]]:
        result_das = []
        for query in filter_queries:
            result_das.append(self._filter(query, limit))
        return result_das

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:

        search_query = {
            'bool': {
                'must': [
                    {'match': {search_field: query}},
                ],
            }
        }
        body = {
            'size': limit,
            'query': search_query,
        }

        resp = self._client.search(
            index=self._index_name,
            body=body,
        )

        docs, scores = self._format_response(resp)

        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        result_das = []
        result_scores = []

        for query in queries:
            documents, scores = self._text_search(query, limit, search_field)
            result_das.append(documents)
            result_scores.append(scores)

        return _FindResultBatched(documents=result_das, scores=np.array(result_scores))  # type: ignore

    ###############################################
    # Helpers                                     #
    ###############################################

    # ElasticSearch helpers
    def _create_index_mapping(self, col: '_ColumnInfo') -> Dict[str, Any]:
        """Create a new HNSW index for a column, and initialize it."""

        index = {'type': col.config['type'] if 'type' in col.config else col.db_type}

        if col.db_type == 'dense_vector':
            for k in self._index_vector_params:
                index[k] = col.config[k]
            if col.n_dim:
                index['dims'] = col.n_dim

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

    def _format_response(self, response: Any) -> Tuple[List[Dict], List[float]]:
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

        return docs, scores

    def _refresh(self, index_name: str):
        self._client.indices.refresh(index=index_name)
