import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Optional, Sequence, Type, TypeVar, Union

import hnswlib
import numpy as np

import docarray.typing
from docarray import BaseDocument, DocumentArray
from docarray.proto import DocumentProto
from docarray.storage.abstract_doc_store import (
    BaseDocumentStore,
    FindResultBatched,
    _Column,
)
from docarray.typing import AnyTensor
from docarray.utils.find import FindResult
from docarray.utils.misc import torch_imported
from docarray.utils.protocols import IsDataclass

TSchema = TypeVar('TSchema', bound=BaseDocument)

HNSWLIB_PY_VEC_TYPES = [list, tuple, np.ndarray]
if torch_imported:
    import torch

    HNSWLIB_PY_VEC_TYPES.append(torch.Tensor)


@dataclass
class HNSWConfig:
    work_dir: str = '.'


class HnswDocumentStore(BaseDocumentStore, Generic[TSchema]):
    _default_column_config = {
        np.ndarray: {
            'dim': 128,
            'space': 'l2',
            'max_elements': 1024,
            'ef_construction': 200,
            'M': 16,
        },
        None: {},
    }

    def __init__(self, config: Optional[IsDataclass] = None):
        super().__init__(config)
        self._work_dir = config.work_dir
        load_existing = os.path.exists(self._work_dir)
        Path(self._work_dir).mkdir(parents=True, exist_ok=True)

        self._doc_id_to_hnsw_id = {}
        self._hnsw_id_to_doc_id = {}

        # HNSWLib setup
        self._index_construct_params = ('space', 'dim')
        self._index_init_params = ('max_elements', 'ef_construction', 'M')

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
        _create_docs_table(self._sqlite_cursor)
        self._sqlite_conn.commit()

    def _create_index_class(self, col: '_Column') -> hnswlib.Index:
        '''Create an instance of hnswlib.Index without initializing it.'''
        construct_params = dict(
            (k, col.config[k]) for k in self._index_construct_params
        )
        if col.n_dim:
            construct_params['dim'] = col.n_dim
        return hnswlib.Index(**construct_params)

    def _create_index(self, col: '_Column') -> hnswlib.Index:
        '''Create a new HNSW index for a column, and initialize it.'''
        index = self._create_index_class(col)
        init_params = dict((k, col.config[k]) for k in self._index_init_params)
        index.init_index(**init_params)
        return index

    def _load_index(self, col_name: str, col: '_Column') -> hnswlib.Index:
        '''Load an existing HNSW index from disk.'''
        index = self._create_index_class(col)
        index.load_index(self._hnsw_locations[col_name])
        return index

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type."""
        for allowed_type in HNSWLIB_PY_VEC_TYPES:
            if issubclass(python_type, allowed_type):
                return np.ndarray

        if python_type == docarray.typing.ID:
            return None  # TODO(johannes): handle this

        raise ValueError(f'Unsupported column type for {type(self)}: {python_type}')

    def _to_numpy(self, val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return val
        elif isinstance(val, (list, tuple)):
            return np.array(val)
        elif torch_imported and isinstance(val, torch.Tensor):
            return val.numpy()
        else:
            raise ValueError(f'Unsupported input type for {type(self)}: {type(val)}')

    def index(self, docs: Union[TSchema, Sequence[TSchema]]):
        """Index a document into the store"""
        data_by_columns = self.get_data_by_columns(docs)

        num_docs_before = _get_num_docs_sqlite(self._sqlite_cursor)
        hnsw_ids = range(num_docs_before, num_docs_before + len(docs))
        # the dict updates below could maybe be optimized?
        self._doc_id_to_hnsw_id.update(
            (doc_id, hnsw_id) for doc_id, hnsw_id in zip(docs.id, hnsw_ids)
        )
        self._hnsw_id_to_doc_id.update(
            (hnsw_id, doc_id) for doc_id, hnsw_id in zip(docs.id, hnsw_ids)
        )

        # indexing into HNSWLib and SQLite sequentially
        # could be improved by processing in parallel
        for col_name, index in self._hnsw_indices.items():
            data = data_by_columns[col_name]
            data_np = [self._to_numpy(arr) for arr in data]
            data_stacked = np.stack(data_np)
            index.add_items(data_stacked, ids=hnsw_ids)
            index.save_index(self._hnsw_locations[col_name])

        _send_docs_to_sqlite(self._sqlite_cursor, docs)
        self._sqlite_conn.commit()

    def find_batched(
        self,
        query: Union[AnyTensor, DocumentArray],
        embedding_field: str = 'embedding',
        metric: str = 'cosine_sim',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        # the below should be done in the abstract class
        if isinstance(query, BaseDocument):
            query_vec = self.get_value(query, embedding_field)
        else:
            query_vec = query
        query_vec_np = self._to_numpy(query_vec)

        index = self._hnsw_indices[embedding_field]
        labels, distances = index.knn_query(query_vec_np, k=limit)
        result_das = [
            _get_docs_from_sqlite(
                self._sqlite_cursor,
                [self._hnsw_id_to_doc_id[id_] for id_ in ids_per_query],
                self._schema,
            )
            for ids_per_query in labels
        ]
        return FindResultBatched(documents=result_das, scores=distances)

    def find(self, *args, **kwargs):
        batched_result = self.find_batched(*args, **kwargs)
        return FindResult(
            documents=batched_result.documents[0], scores=batched_result.scores[0]
        )

    def __delitem__(self, key: Union[str, Sequence[str]]):
        # delete from the indices
        if isinstance(key, str):
            key = [key]
        for k in key:
            id_ = self._doc_id_to_hnsw_id[k]
            for col_name, index in self._hnsw_indices.items():
                index.mark_deleted(id_)
            del self._doc_id_to_hnsw_id[k]

        _delete_docs_from_sqlite(self._sqlite_cursor, key)
        self._sqlite_conn.commit()


# serialization helpers
def _doc_to_bytes(doc: BaseDocument) -> bytes:
    return doc.to_protobuf().SerializeToString()


TSerialize = TypeVar('TSerialize', bound=BaseDocument)


def _doc_from_bytes(data: bytes, doc_class: Type[TSerialize]) -> TSerialize:
    return doc_class.from_protobuf(DocumentProto.FromString(data))


# SQLite helpers


def _create_docs_table(cursor: sqlite3.Cursor):
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, data BLOB)'
    )


def _send_docs_to_sqlite(cursor: sqlite3.Cursor, docs: Sequence[BaseDocument]):
    cursor.executemany(
        'INSERT INTO docs VALUES (?, ?)',
        ((doc.id, _doc_to_bytes(doc)) for doc in docs),
    )


def _get_docs_from_sqlite(
    cursor: sqlite3.Cursor, doc_ids: Sequence[str], doc_class: Type[TSerialize]
) -> DocumentArray:
    cursor.execute(
        'SELECT data FROM docs WHERE doc_id IN (%s)' % ','.join('?' * len(doc_ids)),
        doc_ids,
    )
    rows = cursor.fetchall()
    return DocumentArray[doc_class](
        (_doc_from_bytes(row[0], doc_class) for row in rows)
    )


def _delete_docs_from_sqlite(cursor: sqlite3.Cursor, doc_ids: Sequence[str]):
    cursor.execute(
        'DELETE FROM docs WHERE doc_id IN (%s)' % ','.join('?' * len(doc_ids)),
        doc_ids,
    )


def _get_num_docs_sqlite(cursor: sqlite3.Cursor) -> int:
    cursor.execute('SELECT COUNT(*) FROM docs')
    return cursor.fetchone()[0]
