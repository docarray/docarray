import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
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

import hnswlib
import numpy as np
from elasticsearch import Elasticsearch

import docarray.typing
from docarray import BaseDocument, DocumentArray
from docarray.doc_index.abstract_doc_index import (
    BaseDocumentIndex,
    FindResultBatched,
    _Column,
)
from docarray.utils.find import FindResult
from docarray.utils.misc import torch_imported

# mypy: ignore-errors

if TYPE_CHECKING:

    def composable(fn):  # type: ignore
        return fn

else:
    # static type checkers do not like callable objects as decorators
    from docarray.doc_index.abstract_doc_index import composable

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='ElasticDocumentIndex')

ELASTIC_PY_VEC_TYPES = [list, tuple, np.ndarray]
if torch_imported:
    import torch

    ELASTIC_PY_VEC_TYPES.append(torch.Tensor)


class ElasticDocumentIndex(BaseDocumentIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):

        if db_config.index_name is None:
            id = uuid.uuid4().hex
            db_config.index_name = 'index__' + id

        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(ElasticDocumentIndex.DBConfig, self._db_config)

        self._client = Elasticsearch(
            hosts=self._db_config.hosts,
            **self._db_config.es_config,
        )

        # ElasticSearh index setup
        self._index_init_params = ('dims', 'similarity', 'type')
        self._index_options = (
            'm',
            'ef_construction',
        )

        self._hnsw_indices = {}
        for col_name, col in self._columns.items():
            if not col.config:
                continue  # do not create column index if no config is given
            self._hnsw_indices[col_name] = self._create_index(col)

        mappings = {'dynamic': 'true', '_source': {'enabled': 'true'}, 'properties': {}}
        for col_name, index in self._hnsw_indices.items():
            mappings['properties'][col_name] = index

        if self._client.indices.exists(index=self._db_config.index_name):
            self._client.indices.put_mapping(
                index=self._db_config.index_name, put_mapping=mappings['properties']
            )
        else:
            self._client.indices.create(
                index=self._db_config.index_name, mappings=mappings
            )

        self._client.indices.refresh(index=self._config.index_name)

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################
    class QueryBuilder(BaseDocumentIndex.QueryBuilder):
        def build(self, *args, **kwargs) -> Any:
            return self._queries

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):
        hosts: Union[
            str, List[Union[str, Mapping[str, Union[str, int]]]], None
        ] = 'http://localhost:9200'
        index_name: Optional[str] = None
        es_config: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {
                    'type': 'dense_vector',
                    'dim': 128,
                    'similarity': 'cosine',  # 'l2_norm', 'dot_product', 'cosine'
                    'm': 16,
                    'ef_construction': 100,
                },
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
                return list

        if python_type == docarray.typing.ID:
            return None  # TODO(johannes): handle this

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _index(self, column_data_dic, **kwargs):
        ...

    def index(self, docs: Union[BaseDocument, Sequence[BaseDocument]], **kwargs):
        """Index a document into the store"""
        ...

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        ...

    def _find_batched(
        self,
        query: np.ndarray,
        search_field: str,
        limit: int,
        **kwargs,
    ) -> FindResultBatched:
        ...

    @composable
    def _find(
        self, query: np.ndarray, search_field: str, limit: int, **kwargs
    ) -> FindResult:
        ...

    @composable
    def _filter(
        self,
        *args,
        **kwargs,
    ) -> DocumentArray:
        ...

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
        **kwargs,
    ) -> List[DocumentArray]:
        ...

    def _text_search(
        self,
        query: str,
        search_field: str,
        limit: int,
        **kwargs,
    ) -> FindResult:
        ...

    def _text_search_batched(
        self,
        queries: Sequence[str],
        search_field: str,
        limit: int,
        **kwargs,
    ) -> FindResultBatched:
        ...

    ####################################################
    # Optional overrides                               #
    ####################################################

    ###############################################
    # Helpers                                     #
    ###############################################

    # general helpers
    @staticmethod

    # ElasticSearch helpers
    def _create_index(self, col: '_Column') -> hnswlib.Index:
        """Create a new HNSW index for a column, and initialize it."""
        index = dict((k, col.config[k]) for k in self._index_init_params)
        if col.n_dim:
            index['dims'] = col.n_dim
        index['index_options'] = dict((k, col.config[k]) for k in self._index_options)
        return index
