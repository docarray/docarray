import glob
import hashlib
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from docarray import BaseDoc, DocList
from docarray.array.any_array import AnyDocArray
from docarray.index.abstract import (
    BaseDocIndex,
    _ColumnInfo,
    _raise_not_composable,
    _raise_not_supported,
)
from docarray.index.backends.helper import _collect_query_args
from docarray.proto import DocProto
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils._internal.misc import import_library, is_np_int
from docarray.utils.filter import filter_docs
from docarray.utils.find import FindResult, _FindResult, _FindResultBatched

if TYPE_CHECKING:
    import hnswlib
    import tensorflow as tf  # type: ignore
    import torch

    from docarray.typing import TensorFlowTensor
else:
    hnswlib = import_library('hnswlib', raise_error=True)
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

OPERATOR_MAPPING = {
    '$eq': '=',
    '$neq': '!=',
    '$lt': '<',
    '$lte': '<=',
    '$gt': '>',
    '$gte': '>=',
}


class HnswDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        """Initialize HnswDocumentIndex"""
        if db_config is not None and getattr(db_config, 'index_name'):
            db_config.work_dir = db_config.index_name.replace("__", "/")

        super().__init__(db_config=db_config, **kwargs)
        self._db_config = cast(HnswDocumentIndex.DBConfig, self._db_config)
        self._work_dir = self._db_config.work_dir
        self._logger.debug(f'Working directory set to {self._work_dir}')
        load_existing = os.path.exists(self._work_dir) and glob.glob(
            f'{self._work_dir}/*.bin'
        )
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
            if safe_issubclass(col.docarray_type, AnyDocArray):
                continue
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
        self._column_names: List[str] = []
        self._create_docs_table()
        self._sqlite_conn.commit()
        self._num_docs = 0  # recompute again when needed
        self._logger.info(f'{self.__class__.__name__} has been initialized')

    @property
    def index_name(self):
        return self._db_config.work_dir  # type: ignore

    @property
    def out_schema(self) -> Type[BaseDoc]:
        """Return the real schema of the index."""
        if self._is_subindex:
            return self._ori_schema
        return cast(Type[BaseDoc], self._schema)

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
        """Dataclass that contains all "static" configurations of HnswDocumentIndex."""

        work_dir: str = '.'
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: defaultdict(
                dict,
                {
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
                },
            )
        )

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of HnswDocumentIndex."""

        pass

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
            if safe_issubclass(python_type, allowed_type):
                return np.ndarray

        # types allowed for filtering
        type_map = {
            int: 'INTEGER',
            float: 'REAL',
            str: 'TEXT',
        }
        for py_type, sqlite_type in type_map.items():
            if safe_issubclass(python_type, py_type):
                return sqlite_type

        return None  # all types allowed, but no db type needed

    def _index(
        self,
        column_to_data: Dict[str, Generator[Any, None, None]],
        docs_validated: Sequence[BaseDoc] = [],
    ):
        self._index_subindex(column_to_data)

        # not needed, we implement `index` directly
        hashed_ids = tuple(self._to_hashed_id(doc.id) for doc in docs_validated)
        # indexing into HNSWLib and SQLite sequentially
        # could be improved by processing in parallel
        for col_name, index in self._hnsw_indices.items():
            data = column_to_data[col_name]
            data_np = [self._to_numpy(arr) for arr in data]
            data_stacked = np.stack(data_np)
            num_docs_to_index = len(hashed_ids)
            index_max_elements = index.get_max_elements()
            current_elements = index.get_current_count()
            if current_elements + num_docs_to_index > index_max_elements:
                new_capacity = max(
                    index_max_elements, current_elements + num_docs_to_index
                )
                self._logger.info(f'Resizing the index to {new_capacity}')
                index.resize_index(new_capacity)
            index.add_items(data_stacked, ids=hashed_ids)
            index.save_index(self._hnsw_locations[col_name])

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

        n_docs = 1 if isinstance(docs, BaseDoc) else len(docs)
        self._logger.debug(f'Indexing {n_docs} documents')
        docs_validated = self._validate_docs(docs)
        self._update_subindex_data(docs_validated)
        data_by_columns = self._get_col_value_dict(docs_validated)

        self._index(data_by_columns, docs_validated, **kwargs)

        self._send_docs_to_sqlite(docs_validated)
        self._sqlite_conn.commit()
        self._num_docs = 0  # recompute again when needed

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        """
        Execute a query on the HnswDocumentIndex.

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

        return self._execute_find_and_filter_query(query)

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        return self._search_and_filter(
            queries=queries, limit=limit, search_field=search_field
        )

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
        rows = self._execute_filter(filter_query=filter_query, limit=limit)
        return DocList[self.out_schema](self._doc_from_bytes(blob) for _, blob in rows)  # type: ignore[name-defined]

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
        for field_name, type_, _ in self._flatten_schema(
            cast(Type[BaseDoc], self._schema)
        ):
            if safe_issubclass(type_, AnyDocArray):
                for id in doc_ids:
                    doc = self.__getitem__(id)
                    sub_ids = [sub_doc.id for sub_doc in getattr(doc, field_name)]
                    del self._subindices[field_name][sub_ids]

        try:
            for doc_id in doc_ids:
                id_ = self._to_hashed_id(doc_id)
                for col_name, index in self._hnsw_indices.items():
                    index.mark_deleted(id_)
        except RuntimeError:
            raise KeyError(f'No document with id {doc_ids} found')

        self._delete_docs_from_sqlite(doc_ids)
        self._sqlite_conn.commit()
        self._num_docs = 0  # recompute again when needed

    def _get_items(self, doc_ids: Sequence[str], out: bool = True) -> Sequence[TSchema]:
        """Get Documents from the hnswlib index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :param out: return the documents in the original schema(True) or inner schema(False) for subindex
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`. Duplicate `doc_ids` can be omitted in the output.
        """
        out_docs = self._get_docs_sqlite_doc_id(doc_ids, out)
        if len(out_docs) == 0:
            raise KeyError(f'No document with id {doc_ids} found')
        return out_docs

    def _doc_exists(self, doc_id: str) -> bool:
        hash_id = self._to_hashed_id(doc_id)
        self._sqlite_cursor.execute(f"SELECT data FROM docs WHERE doc_id = '{hash_id}'")
        rows = self._sqlite_cursor.fetchall()
        return len(rows) > 0

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        if self._num_docs == 0:
            self._num_docs = self._get_num_docs_sqlite()
        return self._num_docs

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
        columns: List[Tuple[str, str]] = []
        for col, info in self._column_infos.items():
            if (
                col == 'id'
                or '__' in col
                or not info.db_type
                or info.db_type == np.ndarray
            ):
                continue
            columns.append((col, info.db_type))

        columns_str = ', '.join(f'{name} {type}' for name, type in columns)
        if columns_str:
            columns_str = ', ' + columns_str

        query = f'CREATE TABLE IF NOT EXISTS docs (doc_id INTEGER PRIMARY KEY, data BLOB{columns_str})'
        self._sqlite_cursor.execute(query)

    def _send_docs_to_sqlite(self, docs: Sequence[BaseDoc]):
        # Generate the IDs
        ids = (self._to_hashed_id(doc.id) for doc in docs)

        column_names = self._get_column_names()
        # Construct the field names and placeholders for the SQL query
        all_fields = ', '.join(column_names)
        placeholders = ', '.join(['?'] * len(column_names))

        # Prepare the SQL statement
        query = f'INSERT OR REPLACE INTO docs ({all_fields}) VALUES ({placeholders})'

        # Prepare the data for insertion
        data_to_insert = (
            (id_, self._doc_to_bytes(doc))
            + tuple(getattr(doc, field) for field in column_names[2:])
            for id_, doc in zip(ids, docs)
        )

        # Execute the query
        self._sqlite_cursor.executemany(query, data_to_insert)

    def _get_docs_sqlite_unsorted(self, univ_ids: Sequence[int], out: bool = True):
        for id_ in univ_ids:
            # I hope this protects from injection attacks
            # properly binding with '?' doesn't work for some reason
            assert isinstance(id_, int) or is_np_int(id_)
        sql_id_list = '(' + ', '.join(str(id_) for id_ in univ_ids) + ')'
        self._sqlite_cursor.execute(
            'SELECT data FROM docs WHERE doc_id IN %s' % sql_id_list,
        )
        rows = self._sqlite_cursor.fetchall()
        schema = self.out_schema if out else self._schema
        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], schema))
        return docs_cls([self._doc_from_bytes(row[0], out) for row in rows])

    def _get_docs_sqlite_doc_id(
        self, doc_ids: Sequence[str], out: bool = True
    ) -> DocList[TSchema]:
        hashed_ids = tuple(self._to_hashed_id(id_) for id_ in doc_ids)
        docs_unsorted = self._get_docs_sqlite_unsorted(hashed_ids, out)
        schema = self.out_schema if out else self._schema
        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], schema))
        return docs_cls(sorted(docs_unsorted, key=lambda doc: doc_ids.index(doc.id)))

    def _get_docs_sqlite_hashed_id(self, hashed_ids: Sequence[int]) -> DocList:
        docs_unsorted = self._get_docs_sqlite_unsorted(hashed_ids)

        def _in_position(doc):
            return hashed_ids.index(self._to_hashed_id(doc.id))

        docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self.out_schema))
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

    def _doc_from_bytes(self, data: bytes, out: bool = True) -> BaseDoc:
        schema = self.out_schema if out else self._schema
        schema_cls = cast(Type[BaseDoc], schema)
        return schema_cls.from_protobuf(DocProto.FromString(data))

    def _get_root_doc_id(self, id: str, root: str, sub: str) -> str:
        """Get the root_id given the id of a subindex Document and the root and subindex name for hnswlib.

        :param id: id of the subindex Document
        :param root: root index name
        :param sub: subindex name
        :return: the root_id of the Document
        """
        subindex = self._subindices[root]

        if not sub:
            sub_doc = subindex._get_items([id], out=False)  # type: ignore
            parent_id = (
                sub_doc[0]['parent_id']
                if isinstance(sub_doc[0], dict)
                else sub_doc[0].parent_id
            )
            return parent_id
        else:
            fields = sub.split('__')
            cur_root_id = subindex._get_root_doc_id(
                id, fields[0], '__'.join(fields[1:])
            )
            return self._get_root_doc_id(cur_root_id, root, '')

    def _get_column_names(self) -> List[str]:
        """
        Retrieves the column names of the 'docs' table in the SQLite database.
        The column names are cached in `self._column_names` to prevent multiple queries to the SQLite database.

        :return: A list of strings, where each string is a column name.
        """
        if not self._column_names:
            self._sqlite_cursor.execute('PRAGMA table_info(docs)')
            info = self._sqlite_cursor.fetchall()
            self._column_names = [row[1] for row in info]
        return self._column_names

    def _search_and_filter(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
        hashed_ids: Optional[Set[str]] = None,
    ) -> _FindResultBatched:
        """
        Executes a search and filter operation on the database.

        :param queries: A numpy array of queries.
        :param limit: The maximum number of results to return.
        :param search_field: The field to search in.
        :param hashed_ids: A set of hashed IDs to filter the results with.
        :return: An instance of _FindResultBatched, containing the matching
            documents and their corresponding scores.
        """
        # If there are no documents or hashed_ids is an empty set, return an empty _FindResultBatched
        if hashed_ids is not None and len(hashed_ids) == 0:
            return _FindResultBatched(documents=[], scores=[])  # type: ignore

        # Set limit as the minimum of the provided limit and the total number of documents
        limit = min(limit, self.num_docs())

        # Ensure the search field is in the HNSW indices
        if search_field not in self._hnsw_indices:
            raise ValueError(
                f'Search field {search_field} is not present in the HNSW indices'
            )

        index = self._hnsw_indices[search_field]

        def accept_hashed_ids(id):
            """Accepts IDs that are in hashed_ids."""
            return id in hashed_ids  # type: ignore[operator]

        # Choose the appropriate filter function based on whether hashed_ids was provided
        extra_kwargs = {'filter': accept_hashed_ids} if hashed_ids else {}

        # If hashed_ids is provided, k is the minimum of limit and the length of hashed_ids; else it is limit
        k = min(limit, len(hashed_ids)) if hashed_ids else limit
        try:
            labels, distances = index.knn_query(queries, k=k, **extra_kwargs)
        except RuntimeError:
            k = min(k, self.num_docs())
            labels, distances = index.knn_query(queries, k=k, **extra_kwargs)

        result_das = [
            self._get_docs_sqlite_hashed_id(
                ids_per_query.tolist(),
            )
            for ids_per_query in labels
        ]

        return _FindResultBatched(documents=result_das, scores=distances)

    @classmethod
    def _build_filter_query(
        cls, query: Union[Dict, str], param_values: List[Any]
    ) -> str:
        """
        Builds a filter query for database operations.

        :param query: Query for filtering.
        :param param_values: A list to store the parameters for the query.
        :return: A string representing a SQL filter query.
        """
        if not isinstance(query, dict):
            raise ValueError('Invalid query')

        if len(query) != 1:
            raise ValueError('Each nested dict must have exactly one key')

        key, value = next(iter(query.items()))

        if key in ['$and', '$or']:
            # Combine subqueries using the AND or OR operator
            subqueries = [cls._build_filter_query(q, param_values) for q in value]
            return f'({f" {key[1:].upper()} ".join(subqueries)})'
        elif key == '$not':
            # Negate the query
            return f'NOT {cls._build_filter_query(value, param_values)}'
        else:  # normal field
            field = key
            if not isinstance(value, dict) or len(value) != 1:
                raise ValueError(f'Invalid condition for field {field}')
            operator_key, operator_value = next(iter(value.items()))

            if operator_key == "$exists":
                # Check for the existence or non-existence of a field
                if operator_value:
                    return f'{field} IS NOT NULL'
                else:
                    return f'{field} IS NULL'
            elif operator_key not in OPERATOR_MAPPING:
                raise ValueError(f"Invalid operator {operator_key}")
            else:
                # If the operator is valid, create a placeholder and append the value to param_values
                operator = OPERATOR_MAPPING[operator_key]
                placeholder = '?'
                param_values.append(operator_value)
                return f'{field} {operator} {placeholder}'

    def _execute_filter(
        self,
        filter_query: Any,
        limit: int,
    ) -> List[Tuple[str, bytes]]:
        """
        Executes a filter query on the database.

        :param filter_query: Query for filtering.
        :param limit: Maximum number of rows to be fetched.
        :return: A list of rows fetched from the database.
        """
        param_values: List[Any] = []
        sql_query = self._build_filter_query(filter_query, param_values)
        sql_query = f'SELECT doc_id, data FROM docs WHERE {sql_query} LIMIT {limit}'
        return self._sqlite_cursor.execute(sql_query, param_values).fetchall()

    def _execute_find_and_filter_query(
        self, query: List[Tuple[str, Dict]]
    ) -> FindResult:
        """
        Executes a query to find and filter documents.

        :param query: A list of operations and their corresponding arguments.
        :return: A FindResult object containing filtered documents and their scores.
        """
        # Dictionary to store the score of each document
        doc_to_score: Dict[BaseDoc, Any] = {}

        # Pre- and post-filter conditions
        pre_filters: Dict[str, Dict] = {}
        post_filters: Dict[str, Dict] = {}

        # Define filter limits
        pre_filter_limit = self.num_docs()
        post_filter_limit = self.num_docs()

        find_executed: bool = False

        # Document list with output schema
        out_docs: DocList = DocList[self.out_schema]()  # type: ignore[name-defined]

        for op, op_kwargs in query:
            if op == 'find':
                hashed_ids: Optional[Set[str]] = None
                if pre_filters:
                    hashed_ids = self._pre_filtering(pre_filters, pre_filter_limit)

                query_vector = self._get_vector_for_query_builder(op_kwargs)
                # Perform search and filter if hashed_ids returned by pre-filtering is not empty
                if not (pre_filters and not hashed_ids):
                    # Returns batched output, so we need to get the first lists
                    out_docs, scores = self._search_and_filter(  # type: ignore[assignment]
                        queries=query_vector,
                        limit=op_kwargs.get('limit', self.num_docs()),
                        search_field=op_kwargs['search_field'],
                        hashed_ids=hashed_ids,
                    )
                    out_docs = DocList[self.out_schema](out_docs[0])  # type: ignore[name-defined]
                    doc_to_score.update(zip(out_docs.__getattribute__('id'), scores[0]))
                find_executed = True
            elif op == 'filter':
                if find_executed:
                    post_filters, post_filter_limit = self._update_filter_conditions(
                        post_filters, op_kwargs, post_filter_limit
                    )
                else:
                    pre_filters, pre_filter_limit = self._update_filter_conditions(
                        pre_filters, op_kwargs, pre_filter_limit
                    )
            else:
                raise ValueError(f'Query operation is not supported: {op}')

        if post_filters:
            out_docs = self._post_filtering(
                out_docs, post_filters, post_filter_limit, find_executed
            )

        return self._prepare_out_docs(out_docs, doc_to_score)

    def _update_filter_conditions(
        self, filter_conditions: Dict, operation_args: Dict, filter_limit: int
    ) -> Tuple[Dict, int]:
        """
        Updates filter conditions based on the operation arguments and updates the filter limit.

        :param filter_conditions: Current filter conditions.
        :param operation_args: Arguments of the operation to be executed.
        :param filter_limit: Current filter limit.
        :return: Updated filter conditions and filter limit.
        """
        # Use '$and' operator if filter_conditions is not empty, else use operation_args['filter_query']
        updated_filter_conditions = (
            {'$and': {**filter_conditions, **operation_args['filter_query']}}
            if filter_conditions
            else operation_args['filter_query']
        )
        # Update filter limit based on the operation_args limit
        updated_filter_limit = min(
            filter_limit, operation_args.get('limit', filter_limit)
        )
        return updated_filter_conditions, updated_filter_limit

    def _pre_filtering(
        self, pre_filters: Dict[str, Dict], pre_filter_limit: int
    ) -> Set[str]:
        """
        Performs pre-filtering on the data.

        :param pre_filters: Filter conditions.
        :param pre_filter_limit: Limit for the filtering.
        :return: A set of hashed IDs from the filtered rows.
        """
        rows = self._execute_filter(filter_query=pre_filters, limit=pre_filter_limit)
        return set(hashed_id for hashed_id, _ in rows)

    def _get_vector_for_query_builder(self, find_args: Dict[str, Any]) -> np.ndarray:
        """
        Prepares the query vector for search operation.

        :param find_args: Arguments for the 'find' operation.
        :return: A numpy array representing the query vector.
        """
        if isinstance(find_args['query'], BaseDoc):
            query_vec = self._get_values_by_column(
                [find_args['query']], find_args['search_field']
            )[0]
        else:
            query_vec = find_args['query']
        query_vec_np = self._to_numpy(query_vec)
        query_batched = np.expand_dims(query_vec_np, axis=0)
        return query_batched

    def _post_filtering(
        self,
        out_docs: DocList,
        post_filters: Dict[str, Dict],
        post_filter_limit: int,
        find_executed: bool,
    ) -> DocList:
        """
        Performs post-filtering on the found documents.

        :param out_docs: The documents found by the 'find' operation.
        :param post_filters: The post-filter conditions.
        :param post_filter_limit: Limit for the post-filtering.
        :param find_executed: Whether 'find' operation was executed.
        :return: Filtered documents as per the post-filter conditions.
        """
        if not find_executed:
            out_docs = self.filter(post_filters, limit=self.num_docs())
        else:
            docs_cls = DocList.__class_getitem__(cast(Type[BaseDoc], self.out_schema))
            out_docs = docs_cls(filter_docs(out_docs, post_filters))

        if post_filters:
            out_docs = out_docs[:post_filter_limit]

        return out_docs

    def _prepare_out_docs(
        self, out_docs: DocList, doc_to_score: Dict[BaseDoc, Any]
    ) -> FindResult:
        """
        Prepares output documents with their scores.

        :param out_docs: The documents to be output.
        :param doc_to_score: Mapping of documents to their scores.
        :return: FindResult object with documents and their scores.
        """
        if out_docs:
            # If the "find" operation isn't called through the query builder,
            # all returned scores will be 0
            docs_and_scores = zip(
                out_docs, (doc_to_score.get(doc.id, 0) for doc in out_docs)
            )
            docs_sorted = sorted(docs_and_scores, key=lambda x: x[1])
            out_docs, out_scores = zip(*docs_sorted)
        else:
            out_docs, out_scores = [], []  # type: ignore[assignment]

        return FindResult(documents=out_docs, scores=out_scores)
