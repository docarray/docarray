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
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import hnswlib
import numpy as np

import docarray.typing
from docarray import BaseDocument, DocumentArray
from docarray.proto import DocumentProto
from docarray.storage.abstract_doc_store import (
    BaseDocumentIndex,
    FindResultBatched,
    _Column,
)
from docarray.utils.filter import filter as da_filter
from docarray.utils.find import FindResult
from docarray.utils.misc import is_np_int, torch_imported

if TYPE_CHECKING:

    def composable(fn):  # type: ignore
        return fn

else:
    # static type checkers do not like callable objects as decorators
    from docarray.storage.abstract_doc_store import composable

TSchema = TypeVar('TSchema', bound=BaseDocument)
T = TypeVar('T', bound='HnswDocumentIndex')

HNSWLIB_PY_VEC_TYPES = [list, tuple, np.ndarray]
if torch_imported:
    import torch

    HNSWLIB_PY_VEC_TYPES.append(torch.Tensor)


class HnswDocumentIndex(BaseDocumentIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        super().__init__(db_config=db_config, **kwargs)
        self._work_dir = self._db_config.work_dir
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
            for col_name, col in self._columns.items()
            if col.config
        }
        self._hnsw_indices = {}
        for col_name, col in self._columns.items():
            if not col.config:
                continue  # do not create column index if no config is given
            if load_existing:
                self._hnsw_indices[col_name] = self._load_index(col_name, col)
            else:
                self._hnsw_indices[col_name] = self._create_index(col)

        # SQLite setup
        self._sqlite_db_path = os.path.join(self._work_dir, 'docs_sqlite.db')
        self._sqlite_conn = sqlite3.connect(self._sqlite_db_path)
        self._sqlite_cursor = self._sqlite_conn.cursor()
        self._create_docs_table()
        self._sqlite_conn.commit()

    ###############################################
    # Inner classes for query builder and configs #
    ###############################################
    class QueryBuilder(BaseDocumentIndex.QueryBuilder):
        def build(self, *args, **kwargs) -> Any:
            return self._queries

    @dataclass
    class DBConfig(BaseDocumentIndex.DBConfig):
        work_dir: str = '.'

    @dataclass
    class RuntimeConfig(BaseDocumentIndex.RuntimeConfig):
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: {
                np.ndarray: {
                    'dim': 128,
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
        """Map python type to database type."""
        for allowed_type in HNSWLIB_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return np.ndarray

        if python_type == docarray.typing.ID:
            return None  # TODO(johannes): handle this

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _index(self, column_data_dic, **kwargs):
        # not needed, we implement `index` directly
        ...

    def index(self, docs: Union[BaseDocument, Sequence[BaseDocument]], **kwargs):
        """Index a document into the store"""
        if kwargs:
            raise ValueError(f'{list(kwargs.keys())} are not valid keyword arguments')
        doc_seq = docs if isinstance(docs, Sequence) else [docs]
        data_by_columns = self._get_col_value_dict(doc_seq)
        hashed_ids = tuple(self._to_hashed_id(doc.id) for doc in doc_seq)

        # indexing into HNSWLib and SQLite sequentially
        # could be improved by processing in parallel
        for col_name, index in self._hnsw_indices.items():
            data = data_by_columns[col_name]
            data_np = [self._to_numpy(arr) for arr in data]
            data_stacked = np.stack(data_np)
            index.add_items(data_stacked, ids=hashed_ids)
            index.save_index(self._hnsw_locations[col_name])

        self._send_docs_to_sqlite(doc_seq)
        self._sqlite_conn.commit()

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        if args or kwargs:
            raise ValueError(
                f'args and kwargs not supported for `execute_query` on {type(self)}'
            )

        ann_docs = DocumentArray.__class_getitem__(
            cast(Type[BaseDocument], self._schema)
        )([])
        filter_conditions = []
        doc_to_score: Dict[BaseDocument, Any] = {}
        for op, op_kwargs in query:
            if op == 'find':
                docs, scores = self.find(**op_kwargs)
                ann_docs.extend(docs)
                doc_to_score.update(zip(docs.__getattribute__('id'), scores))
            elif op == 'filter':
                filter_conditions.append(op_kwargs['filter_query'])

        docs_filtered = ann_docs
        for cond in filter_conditions:
            da_cls = DocumentArray.__class_getitem__(
                cast(Type[BaseDocument], self._schema)
            )
            docs_filtered = da_cls(da_filter(docs_filtered, cond))

        docs_and_scores = zip(
            docs_filtered, (doc_to_score[doc.id] for doc in docs_filtered)
        )
        docs_sorted = sorted(docs_and_scores, key=lambda x: x[1])
        out_docs, out_scores = zip(*docs_sorted)
        return FindResult(documents=out_docs, scores=out_scores)

    def _find_batched(
        self,
        query: np.ndarray,
        search_field: str,
        limit: int,
        **kwargs,
    ) -> FindResultBatched:
        if kwargs:
            raise ValueError(f'{list(kwargs.keys())} are not valid keyword arguments')
        index = self._hnsw_indices[search_field]
        labels, distances = index.knn_query(query, k=limit)
        result_das = [
            self._get_docs_sqlite_hashed_id(
                ids_per_query.tolist(),
            )
            for ids_per_query in labels
        ]
        return FindResultBatched(documents=result_das, scores=distances)

    @composable
    def _find(
        self, query: np.ndarray, search_field: str, limit: int, **kwargs
    ) -> FindResult:
        query_batched = np.expand_dims(query, axis=0)
        docs, scores = self._find_batched(query_batched, search_field, limit, **kwargs)
        return FindResult(documents=docs[0], scores=scores[0])

    @composable
    def _filter(
        self,
        *args,
        **kwargs,
    ) -> DocumentArray:

        raise NotImplementedError(
            f'{type(self)} does not support filter-only queries.'
            f' To perform post-filtering on a query, use'
            f' `build_query()` and `execute_query()`.'
        )

    def _filter_batched(
        self,
        filter_queries: Any,
        limit: int,
        **kwargs,
    ) -> List[DocumentArray]:
        raise NotImplementedError(
            f'{type(self)} does not support filter-only queries.'
            f' To perform post-filtering on a query, use'
            f' `build_query()` and `execute_query()`.'
        )

    def _text_search(
        self,
        query: str,
        search_field: str,
        limit: int,
        **kwargs,
    ) -> FindResult:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _text_search_batched(
        self,
        queries: Sequence[str],
        search_field: str,
        limit: int,
        **kwargs,
    ) -> FindResultBatched:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def __delitem__(self, key: Union[str, Sequence[str]]):
        # delete from the indices
        if isinstance(key, str):
            key = [key]
        try:
            for doc_id in key:
                id_ = self._to_hashed_id(doc_id)
                for col_name, index in self._hnsw_indices.items():
                    index.mark_deleted(id_)
        except RuntimeError:
            raise KeyError(f'No document with id {key} found')

        self._delete_docs_from_sqlite(key)
        self._sqlite_conn.commit()

    def __getitem__(self, key: Union[str, Sequence[str]]):
        # delete from the indices
        if isinstance(key, str):
            key = [key]
        out_docs = self._get_docs_sqlite_doc_id(key)
        # TODO(johannes): this could be handled for all stores
        if len(out_docs) == 0:
            raise KeyError(f'No document with id {key} found')
        if len(out_docs) == 1:
            return out_docs[0]
        return out_docs

    def num_docs(self) -> int:
        return self._get_num_docs_sqlite()

    ###############################################
    # Helpers                                     #
    ###############################################

    # general helpers
    @staticmethod
    def _to_hashed_id(doc_id: str) -> int:
        # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
        # hashing to 18 digits avoids overflow of sqlite INTEGER
        return int(hashlib.sha256(doc_id.encode('utf-8')).hexdigest(), 16) % 10**18

    def _load_index(self, col_name: str, col: '_Column') -> hnswlib.Index:
        """Load an existing HNSW index from disk."""
        index = self._create_index_class(col)
        index.load_index(self._hnsw_locations[col_name])
        return index

    def _to_numpy(self, val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return val
        elif isinstance(val, (list, tuple)):
            return np.array(val)
        elif torch_imported and isinstance(val, torch.Tensor):
            return val.numpy()
        else:
            raise ValueError(f'Unsupported input type for {type(self)}: {type(val)}')

    # HNSWLib helpers
    def _create_index_class(self, col: '_Column') -> hnswlib.Index:
        """Create an instance of hnswlib.Index without initializing it."""
        construct_params = dict(
            (k, col.config[k]) for k in self._index_construct_params
        )
        if col.n_dim:
            construct_params['dim'] = col.n_dim
        return hnswlib.Index(**construct_params)

    def _create_index(self, col: '_Column') -> hnswlib.Index:
        """Create a new HNSW index for a column, and initialize it."""
        index = self._create_index_class(col)
        init_params = dict((k, col.config[k]) for k in self._index_init_params)
        index.init_index(**init_params)
        index.set_ef(col.config['ef'])
        index.set_num_threads(col.config['num_threads'])
        return index

    # SQLite helpers
    def _create_docs_table(self):
        self._sqlite_cursor.execute(
            'CREATE TABLE IF NOT EXISTS docs (doc_id INTEGER PRIMARY KEY, data BLOB)'
        )

    def _send_docs_to_sqlite(self, docs: Sequence[BaseDocument]):
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
        da_cls = DocumentArray.__class_getitem__(cast(Type[BaseDocument], self._schema))
        return da_cls([self._doc_from_bytes(row[0]) for row in rows])

    def _get_docs_sqlite_doc_id(self, doc_ids: Sequence[str]) -> DocumentArray:
        hashed_ids = tuple(self._to_hashed_id(id_) for id_ in doc_ids)
        docs_unsorted = self._get_docs_sqlite_unsorted(hashed_ids)
        da_cls = DocumentArray.__class_getitem__(cast(Type[BaseDocument], self._schema))
        return da_cls(sorted(docs_unsorted, key=lambda doc: doc_ids.index(doc.id)))

    def _get_docs_sqlite_hashed_id(self, hashed_ids: Sequence[int]) -> DocumentArray:
        docs_unsorted = self._get_docs_sqlite_unsorted(hashed_ids)

        def _in_position(doc):
            return hashed_ids.index(self._to_hashed_id(doc.id))

        da_cls = DocumentArray.__class_getitem__(cast(Type[BaseDocument], self._schema))
        return da_cls(sorted(docs_unsorted, key=_in_position))

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
    def _doc_to_bytes(self, doc: BaseDocument) -> bytes:
        return doc.to_protobuf().SerializeToString()

    def _doc_from_bytes(self, data: bytes) -> BaseDocument:
        schema_cls = cast(Type[BaseDocument], self._schema)
        return schema_cls.from_protobuf(DocumentProto.FromString(data))
