from typing import Any, Generic, Optional, Sequence, Type, TypeVar, Union

import hnswlib
import numpy as np

import docarray.typing
from docarray import BaseDocument, DocumentArray
from docarray.storage.abstract_doc_store import BaseDocumentStore, FindResultBatched
from docarray.typing import AnyTensor
from docarray.utils.misc import torch_imported
from docarray.utils.protocols import IsDataclass

TSchema = TypeVar('TSchema', bound=BaseDocument)

HNSWLIB_PY_VEC_TYPES = [list, tuple, np.ndarray]
if torch_imported:
    import torch

    HNSWLIB_PY_VEC_TYPES.append(torch.Tensor)


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
        self._index_construct_params = ('space', 'dim')
        self._index_init_params = ('max_elements', 'ef_construction', 'M')

        self._docs_buffer = DocumentArray[self._schema]()

        self._indices = {}
        for col_name, col in self._columns.items():
            col_config = col.config
            if not col_config:
                continue  # do not create column index if no config is given
            construct_params = dict(
                (k, col_config[k]) for k in self._index_construct_params
            )
            index = hnswlib.Index(**construct_params)
            init_params = dict((k, col_config[k]) for k in self._index_init_params)
            index.init_index(**init_params)
            self._indices[col_name] = index

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
        doc_buf_len_before = len(self._docs_buffer)
        self._docs_buffer.extend(docs)
        for col_name, index in self._indices.items():
            data = data_by_columns[col_name]
            data_np = [self._to_numpy(arr) for arr in data]
            data_stacked = np.stack(data_np)
            index.add_items(
                data_stacked, ids=range(doc_buf_len_before, len(self._docs_buffer))
            )

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

        index = self._indices[embedding_field]
        labels, distances = index.knn_query(query_vec_np, k=limit)
        result_das = [
            DocumentArray[self._schema]([self._docs_buffer[i] for i in ids_per_query])
            for ids_per_query in labels
        ]
        return FindResultBatched(documents=result_das, scores=distances)

    def find(self, *args, **kwargs):
        ...
