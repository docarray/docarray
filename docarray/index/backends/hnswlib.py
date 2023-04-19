import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
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

from docarray import BaseDoc, DocList
from docarray.index.abstract import (
    BaseDocIndex,
    _ColumnInfo,
    _raise_not_composable,
    _raise_not_supported,
)
from docarray.proto import DocProto
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils._internal.misc import import_library, is_np_int
from docarray.utils.filter import filter_docs
from docarray.utils.find import _FindResult, _FindResultBatched

if TYPE_CHECKING:
    import hnswlib
    import tensorflow as tf  # type: ignore
    import torch

    from docarray.typing import TensorFlowTensor
else:
    hnswlib = import_library('hnswlib', raise_error=False)
    torch = import_library('torch', raise_error=False)
    tf = import_library('tensorflow', raise_error=False)
    if tf is not None:
        from docarray.typing import TensorFlowTensor

HNSWLIB_PY_VEC_TYPES: List[Any] = [list, tuple, np.ndarray, AbstractTensor]

if torch is not None:
    HNSWLIB_PY_VEC_TYPES.append(torch.Tensor)  # type: ignore

if tf is not None:
    HNSWLIB_PY_VEC_TYPES.append(tf.Tensor)
    HNSWLIB_PY_VEC_TYPES.append(TensorFlowTensor)


TSchema = TypeVar('TSchema', bound=BaseDoc)
T = TypeVar('T', bound='HnswDocumentIndex')


def _collect_query_args(method_name: str):  # TODO: use partialmethod instead
    def inner(self, *args, **kwargs):
        if args:
            raise ValueError(
                f'Positional arguments are not supported for '
                f'`{type(self)}.{method_name}`.'
                f' Use keyword arguments instead.'
            )
        updated_query = self._queries + [(method_name, kwargs)]
        return type(self)(updated_query)

    return inner


class HnswDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        """Initialize HnswDocumentIndex"""
        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(HnswDocumentIndex.DBConfig, self._db_config)
        self._work_dir = self._db_config.work_dir
        self._logger.debug(f'Working directory set to {self._work_dir}')
        load_existing = os.path.exists(self._work_dir) and os.listdir(self._work_dir)
        Path(self._work_dir).mkdir(parents=True, exist_ok=True)

        # HNSWLib setup
        self._index_construct_params = ('space', 'dim')
        self._index_init_params = (
            'max_elements',
            'ef_construction',
            'M',
            'allow_replace_deleted',
        )

        self._hnsw_locations = {
            col_name: os.path.join(self._work_dir, f'{col_name}.bin')
            for col_name, col in self._column_infos.items()
            if col.config
        }
        self._hnsw_indices = {}
        for col_name, col in self._column_infos.items():
            if not col.config:
                # non-tensor type; don't create an index
                continue
            if not load_existing and (
                (not col.n_dim and col.config['dim'] < 0) or not col.config['index']
            ):
                # tensor type, but don't index
                self._logger.info(
                    f'Not indexing column {col_name}; either `index=False` is set or no dimensionality is specified'
                )
                continue
            if load_existing:
                self._hnsw_indices[col_name] = self._load_index(col_name, col)
                self._logger.info(f'Loading an existing index for column `{col_name}`')
            else:
                self._hnsw_indices[col_name] = self._create_index(col_name, col)
                self._logger.info(f'Created a new index for column `{col_name}`')

        # SQLite setup
        self._sqlite_db_path = os.path.join(self._work_dir, 'docs_sqlite.db')
        self._logger.debug(f'DB path set to {self._sqlite_db_path}')
        self._sqlite_conn = sqlite3.connect(self._sqlite_db_path)
        self._logger.info('Connection to DB has been established')
        self._sqlite_cursor = self._sqlite_conn.cursor()
        self._create_docs_table()
        self._sqlite_conn.commit()
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################
    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, *args, **kwargs) -> Any:
            """Build the query object."""
            return self._queries

        find = _collect_query_args('find')
        filter = _collect_query_args('filter')
        text_search = _raise_not_supported('text_search')
        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('find_batched')
        text_search_batched = _raise_not_supported('text_search')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of WeaviateDocumentIndex."""

        work_dir: str = '.'

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of WeaviateDocumentIndex."""

        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {
                    'dim': -1,
                    'index': True,  # if False, don't index at all
                    'space': 'l2',  # 'l2', 'ip', 'cosine'
                    'max_elements': 1024,
                    'ef_construction': 200,
                    'ef': 10,
                    'M': 16,
                    'allow_replace_deleted': True,
                    'num_threads': 1,
                },
                # `None` is not a Type, but we allow it here anyway
                None: {},  # type: ignore
            }
        )

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
        for allowed_type in HNSWLIB_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return np.ndarray

        return None  # all types allowed, but no db type needed

    def _index(self, column_data_dic, **kwargs):
        # not needed, we implement `index` directly
        ...

    def index(self, docs: Union[BaseDoc, Sequence[BaseDoc]], **kwargs):
        """Index Documents into the index.

        !!! note
            Passing a sequence of Documents that is not a DocList
            (such as a List of Docs) comes at a performance penalty.
            This is because the Index needs to check compatibility between itself and
            the data. With a DocList as input this is a single check; for other inputs
            compatibility needs to be checked for every Document individually.

        :param docs: Documents to index.
        """
        if kwargs:
            raise ValueError(f'{list(kwargs.keys())} are not valid keyword arguments')

        self._logger.debug(f'Indexing {len(docs)} documents')
        docs_validated = self._validate_docs(docs)
        data_by_columns = self._get_col_value_dict(docs_validated)
        hashed_ids = tuple(self._to_hashed_id(doc.id) for doc in docs_validated)
        # indexing into HNSWLib and SQLite sequentially
        # could be improved by processing in parallel
        for col_name, index in self._hnsw_indices.items():
            data = data_by_columns[col_name]
            data_np = [self._to_numpy(arr) for arr in data]
            data_stacked = np.stack(data_np)
            index.add_items(data_stacked, ids=hashed_ids)
            index.save_index(self._hnsw_locations[col_name])

        self._send_docs_to_sqlite(docs_validated)
        self._sqlite_conn.commit()

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        """
        Execute a query on the WeaviateDocumentIndex.

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

        ann_docs = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))([])
        filter_conditions = []
        doc_to_score: Dict[BaseDoc, Any] = {}
        for op, op_kwargs in query:
            if op == 'find':
                docs, scores = self.find(**op_kwargs)
                ann_docs.extend(docs)
                doc_to_score.update(zip(docs.__getattribute__('id'), scores))
            elif op == 'filter':
                filter_conditions.append(op_kwargs['filter_query'])

        self._logger.debug(f'Executing query {query}')
        docs_filtered = ann_docs
        for cond in filter_conditions:
            docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
            docs_filtered = docs_cls(filter_docs(docs_filtered, cond))

        self._logger.debug(f'{len(docs_filtered)} results found')
        docs_and_scores = zip(
            docs_filtered, (doc_to_score[doc.id] for doc in docs_filtered)
        )
        docs_sorted = sorted(docs_and_scores, key=lambda x: x[1])
        out_docs, out_scores = zip(*docs_sorted)
        return _FindResult(documents=out_docs, scores=out_scores)

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        index = self._hnsw_indices[search_field]
        labels, distances = index.knn_query(queries, k=limit)
        result_das = [
            self._get_docs_sqlite_hashed_id(
                ids_per_query.tolist(),
            )
            for ids_per_query in labels
        ]
        return _FindResultBatched(documents=result_das, scores=distances)

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        query_batched = np.expand_dims(query, axis=0)
        docs, scores = self._find_batched(
            queries=query_batched, limit=limit, search_field=search_field
        )
        return _FindResult(
            documents=docs[0], scores=NdArray._docarray_from_native(scores[0])
        )

    def _filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> DocList:
        raise NotImplementedError(
            f'{type(self)} does not support filter-only queries.'
            f' To perform post-filtering on a query, use'
            f' `build_query()` and `execute_query()`.'
        )

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
    ) -> List[DocList]:
        raise NotImplementedError(
            f'{type(self)} does not support filter-only queries.'
            f' To perform post-filtering on a query, use'
            f' `build_query()` and `execute_query()`.'
        )

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _del_items(self, doc_ids: Sequence[str]):
        # delete from the indices
        try:
            for doc_id in doc_ids:
                id_ = self._to_hashed_id(doc_id)
                for col_name, index in self._hnsw_indices.items():
                    index.mark_deleted(id_)
        except RuntimeError:
            raise KeyError(f'No document with id {doc_ids} found')

        self._delete_docs_from_sqlite(doc_ids)
        self._sqlite_conn.commit()

    def _get_items(self, doc_ids: Sequence[str]) -> Sequence[TSchema]:
        out_docs = self._get_docs_sqlite_doc_id(doc_ids)
        if len(out_docs) == 0:
            raise KeyError(f'No document with id {doc_ids} found')
        return out_docs

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        return self._get_num_docs_sqlite()

    ###############################################
    # Helpers                                     #
    ###############################################

    # general helpers
    @staticmethod
    def _to_hashed_id(doc_id: Optional[str]) -> int:
        # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
        # hashing to 18 digits avoids overflow of sqlite INTEGER
        if doc_id is None:
            raise ValueError(
                'The Document id is None. To use DocumentIndex it needs to be set.'
            )
        return int(hashlib.sha256(doc_id.encode('utf-8')).hexdigest(), 16) % 10**18

    def _load_index(self, col_name: str, col: '_ColumnInfo') -> hnswlib.Index:
        """Load an existing HNSW index from disk."""
        index = self._create_index_class(col)
        index.load_index(self._hnsw_locations[col_name])
        return index

    # HNSWLib helpers
    def _create_index_class(self, col: '_ColumnInfo') -> hnswlib.Index:
        """Create an instance of hnswlib.index without initializing it."""
        construct_params = dict(
            (k, col.config[k]) for k in self._index_construct_params
        )
        if col.n_dim:
            construct_params['dim'] = col.n_dim
        return hnswlib.Index(**construct_params)

    def _create_index(self, col_name: str, col: '_ColumnInfo') -> hnswlib.Index:
        """Create a new HNSW index for a column, and initialize it."""
        index = self._create_index_class(col)
        init_params = dict((k, col.config[k]) for k in self._index_init_params)
        index.init_index(**init_params)
        index.set_ef(col.config['ef'])
        index.set_num_threads(col.config['num_threads'])
        index.save_index(self._hnsw_locations[col_name])
        return index

    # SQLite helpers
    def _create_docs_table(self):
        self._sqlite_cursor.execute(
            'CREATE TABLE IF NOT EXISTS docs (doc_id INTEGER PRIMARY KEY, data BLOB)'
        )

    def _send_docs_to_sqlite(self, docs: Sequence[BaseDoc]):
        ids = (self._to_hashed_id(doc.id) for doc in docs)
        self._sqlite_cursor.executemany(
            'INSERT INTO docs VALUES (?, ?)',
            ((id_, self._doc_to_bytes(doc)) for id_, doc in zip(ids, docs)),
        )

    def _get_docs_sqlite_unsorted(self, univ_ids: Sequence[int]):
        for id_ in univ_ids:
            # I hope this protects from injection attacks
            # properly binding with '?' doesn't work for some reason
            assert isinstance(id_, int) or is_np_int(id_)
        sql_id_list = '(' + ', '.join(str(id_) for id_ in univ_ids) + ')'
        self._sqlite_cursor.execute(
            'SELECT data FROM docs WHERE doc_id IN %s' % sql_id_list,
        )
        rows = self._sqlite_cursor.fetchall()
        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
        return docs_cls([self._doc_from_bytes(row[0]) for row in rows])

    def _get_docs_sqlite_doc_id(self, doc_ids: Sequence[str]) -> DocList[TSchema]:
        hashed_ids = tuple(self._to_hashed_id(id_) for id_ in doc_ids)
        docs_unsorted = self._get_docs_sqlite_unsorted(hashed_ids)
        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
        return docs_cls(sorted(docs_unsorted, key=lambda doc: doc_ids.index(doc.id)))

    def _get_docs_sqlite_hashed_id(self, hashed_ids: Sequence[int]) -> DocList:
        docs_unsorted = self._get_docs_sqlite_unsorted(hashed_ids)

        def _in_position(doc):
            return hashed_ids.index(self._to_hashed_id(doc.id))

        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self._schema))
        return docs_cls(sorted(docs_unsorted, key=_in_position))

    def _delete_docs_from_sqlite(self, doc_ids: Sequence[Union[str, int]]):
        ids = tuple(
            self._to_hashed_id(id_) if isinstance(id_, str) else id_ for id_ in doc_ids
        )
        self._sqlite_cursor.execute(
            'DELETE FROM docs WHERE doc_id IN (%s)' % ','.join('?' * len(ids)),
            ids,
        )

    def _get_num_docs_sqlite(self) -> int:
        self._sqlite_cursor.execute('SELECT COUNT(*) FROM docs')
        return self._sqlite_cursor.fetchone()[0]

    # serialization helpers
    def _doc_to_bytes(self, doc: BaseDoc) -> bytes:
        return doc.to_protobuf().SerializeToString()

    def _doc_from_bytes(self, data: bytes) -> BaseDoc:
        schema_cls = cast(Type[BaseDoc], self._schema)
        return schema_cls.from_protobuf(DocProto.FromString(data))
