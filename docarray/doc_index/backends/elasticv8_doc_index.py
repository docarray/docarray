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

import docarray.typing
from docarray import BaseDocument
from docarray.doc_index.abstract_doc_index import (
    BaseDocumentIndex,
    _ColumnInfo,
    _FindResultBatched,
    _raise_not_composable,
)
from docarray.typing import AnyTensor
from docarray.utils.find import _FindResult
from docarray.utils.misc import torch_imported

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='ElasticDocumentV8Index')

ELASTIC_PY_VEC_TYPES = [list, tuple, np.ndarray]
ELASTIC_PY_TYPES = [bool, int, float, str, docarray.typing.ID]
if torch_imported:
    import torch

    ELASTIC_PY_VEC_TYPES.append(torch.Tensor)


class ElasticDocumentV8Index(BaseDocumentIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(ElasticDocumentV8Index.DBConfig, self._db_config)

        if self._db_config.index_name is None:
            id = uuid.uuid4().hex
            self._db_config.index_name = 'index__' + id

        self._index_name = self._db_config.index_name

        self._client = Elasticsearch(
            hosts=self._db_config.hosts,
            **self._db_config.es_config,
        )

        # ElasticSearh index setup
        self._index_init_params = ('type',)
        self._index_vector_params = ('dims', 'similarity', 'index')
        self._index_vector_options = ('m', 'ef_construction')

        mappings: Dict[str, Any] = {
            'dynamic': True,
            '_source': {'enabled': 'true'},
            'properties': {},
        }

        for col_name, col in self._column_infos.items():
            if not col.config:
                continue  # do not create column index if no config is given
            mappings['properties'][col_name] = self._create_index(col)

        if self._client.indices.exists(index=self._index_name):  # type: ignore
            self._client.indices.put_mapping(
                index=self._index_name, properties=mappings['properties']
            )
        else:
            self._client.indices.create(index=self._index_name, mappings=mappings)

        if len(self._db_config.index_settings):
            self._client.indices.put_settings(
                index=self._index_name, settings=self._db_config.index_settings
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
            if len(self._query['query']) == 0:
                del self._query['query']
            elif 'knn' in self._query:
                self._query['knn']['filter'] = self._query['query']
                del self._query['query']

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
            self._query['knn'] = {
                'field': search_field,
                'query_vector': query_vec_np,
                'k': limit,
                'num_candidates': self._outer_instance._runtime_config.default_column_config[
                    np.ndarray
                ][
                    'num_candidates'
                ],
            }
            return self

        # filter accrpts Leaf/Compound query clauses
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html
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
        return self.QueryBuilder(self, **kwargs)  # type: ignore

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):

        hosts: Union[
            str, List[Union[str, Mapping[str, Union[str, int]], NodeConfig]], None
        ] = 'http://localhost:9200'
        index_name: Optional[str] = None
        es_config: Dict[str, Any] = field(default_factory=dict)
        index_settings: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {
                    'type': 'dense_vector',
                    'index': True,
                    'dims': 128,
                    'similarity': 'cosine',  # 'l2_norm', 'dot_product', 'cosine'
                    'm': 16,
                    'ef_construction': 100,
                    'num_candidates': 10000,
                },
                docarray.typing.ID: {'type': 'keyword'},
                bool: {'type': 'boolean'},
                int: {'type': 'integer'},
                float: {'type': 'float'},
                str: {'type': 'text'},
                # `None` is not a Type, but we allow it here anyway
                None: {},  # type: ignore
            }
        )

    ###############################################
    # Implementation of abstract methods          #
    ###############################################

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""
        for allowed_type in ELASTIC_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return np.ndarray

        if python_type in ELASTIC_PY_TYPES:
            return python_type

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _index(
        self,
        column_to_data: Dict[str, Generator[Any, None, None]],
        refresh: bool = True,
    ):

        data = self._transpose_col_value_dict(column_to_data)  # type: ignore
        requests = []

        for row in data:
            request = {
                '_index': self._index_name,
                '_id': row['id'],
            }
            for col_name, col in self._column_infos.items():
                if not col.config:
                    continue
                if col.db_type == np.ndarray and np.all(row[col_name] == 0):
                    row[col_name] = row[col_name] + 1.0e-9
                request[col_name] = row[col_name]
            requests.append(request)

        _, warning_info = self._send_requests(requests)
        for info in warning_info:
            warnings.warn(str(info))

        if refresh:
            self._refresh(self._index_name)

    def num_docs(self) -> int:
        return self._client.count(index=self._index_name)['count']

    def _del_items(self, doc_ids: Sequence[str]):
        requests = []
        for _id in doc_ids:
            requests.append(
                {'_op_type': 'delete', '_index': self._index_name, '_id': _id}
            )

        _, warning_info = self._send_requests(requests)

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
            ids=doc_ids,  # type: ignore
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

    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        knn_query = {
            'field': search_field,
            'query_vector': query,
            'k': limit,
            'num_candidates': self._runtime_config.default_column_config[np.ndarray][
                'num_candidates'
            ],
        }

        resp = self._client.search(
            index=self._index_name,
            knn=knn_query,
            size=limit,
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
        resp = self._client.search(
            index=self._index_name,
            query=filter_query,
            size=limit,
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
            "bool": {
                "must": [
                    {"match": {search_field: query}},
                ],
            }
        }

        resp = self._client.search(
            index=self._index_name,
            query=search_query,
            size=limit,
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

        return _FindResultBatched(documents=result_das, scores=np.array(result_scores, dtype=object))  # type: ignore

    def execute_query(self, query: Dict[str, Any], *args, **kwargs) -> Any:
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        resp = self._client.search(index=self._index_name, **query)
        docs, scores = self._format_response(resp)
        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    ###############################################
    # Helpers                                     #
    ###############################################

    # ElasticSearch helpers
    def _create_index(self, col: '_ColumnInfo') -> Dict[str, Any]:
        """Create a new HNSW index for a column, and initialize it."""
        index = dict((k, col.config[k]) for k in self._index_init_params)
        if col.db_type == np.ndarray:
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
        self, request: Iterable[Dict[str, Any]], **kwargs
    ) -> Tuple[List[Dict], List[Any]]:
        """Send bulk request to Elastic and gather the successful info"""

        # TODO chunk_size

        accumulated_info = []
        warning_info = []
        for success, info in parallel_bulk(
            self._client,
            request,
            raise_on_error=False,
            raise_on_exception=False,
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
            doc_dict = result['_source']
            doc_dict['id'] = result['_id']
            docs.append(doc_dict)
            scores.append(result['_score'])

        return docs, scores

    def _refresh(self, index_name: str):
        self._client.indices.refresh(index=index_name)
