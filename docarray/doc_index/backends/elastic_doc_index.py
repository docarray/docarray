import uuid
import warnings
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
)
from docarray.utils.find import _FindResult
from docarray.utils.misc import torch_imported

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='ElasticDocumentIndex')

MAX_ES_RETURNED_DOCS = 10000

ELASTIC_PY_VEC_TYPES = [list, tuple, np.ndarray]
ELASTIC_PY_TYPES = [bool, int, float, str, docarray.typing.ID]
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

        self._refresh(self._index_name)

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################
    # TODO add class QueryBuilder

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):

        hosts: Union[
            str, List[Union[str, Mapping[str, Union[str, int]], NodeConfig]], None
        ] = 'http://localhost:9200'
        index_name: Optional[str] = None
        es_config: Dict[str, Any] = field(default_factory=dict)

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

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):

        data = self._transpose_col_value_dict(column_to_data)  # type: ignore
        requests = []

        for row in data:
            request = {
                '_index': self._index_name,
                '_id': row['id'],
            }
            # TODO change here when more types are supported
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

        self._refresh(self._index_name)  # TODO add runtime config for efficient refresh

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

        for pos in range(0, len(doc_ids), MAX_ES_RETURNED_DOCS):

            es_rows = self._client.mget(
                index=self._index_name,
                ids=doc_ids[pos : pos + MAX_ES_RETURNED_DOCS],  # type: ignore
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

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        ...

    def _find(self, query: np.ndarray, search_field: str, limit: int) -> _FindResult:
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
        )

        docs = []
        scores = []
        for result in resp['hits']['hits']:
            doc_dict = result['_source']
            doc_dict['id'] = result['_id']
            docs.append(doc_dict)
            scores.append(result['_score'])

        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    def _find_batched(
        self,
        queries: np.ndarray,
        search_field: str,
        limit: int,
    ) -> _FindResultBatched:
        result_das = []
        result_scores = []

        for query in queries:
            documents, scores = self._find(query, search_field, limit)
            result_das.append(documents)
            result_scores.append(scores)

        return _FindResultBatched(documents=result_das, scores=np.array(result_scores))  # type: ignore

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> List[Dict]:
        resp = self._client.search(
            index=self._index_name,
            query=filter_query,
            size=limit,
        )

        docs = []
        for result in resp['hits']['hits'][:limit]:
            doc_dict = result['_source']
            doc_dict['id'] = result['_id']
            docs.append(doc_dict)

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
        search_field: str,
        limit: int,
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

        docs = []
        scores = []
        for result in resp['hits']['hits']:
            doc_dict = result['_source']
            doc_dict['id'] = result['_id']
            docs.append(doc_dict)
            scores.append(result['_score'])

        return _FindResult(documents=docs, scores=np.array(scores))  # type: ignore

    def _text_search_batched(
        self,
        queries: Sequence[str],
        search_field: str,
        limit: int,
    ) -> _FindResultBatched:
        result_das = []
        result_scores = []

        for query in queries:
            documents, scores = self._text_search(query, search_field, limit)
            result_das.append(documents)
            result_scores.append(scores)

        return _FindResultBatched(documents=result_das, scores=np.array(result_scores))  # type: ignore

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

    def _refresh(self, index_name: str):
        self._client.indices.refresh(index=index_name)
