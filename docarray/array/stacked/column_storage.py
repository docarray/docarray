from collections import ChainMap
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    MutableMapping,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

from docarray.array.array.array import DocumentArray
from docarray.array.stacked.list_advance_indexing import ListAdvanceIndex
from docarray.base_document import BaseDocument
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._typing import is_tensor_union
from docarray.utils.misc import is_tf_available

if TYPE_CHECKING:
    from docarray.array.stacked.array_stacked import DocumentArrayStacked

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing import TensorFlowTensor
else:
    TensorFlowTensor = None  # type: ignore

IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


T = TypeVar('T', bound='ColumnStorage')


class ColumnStorage:

    document_type: Type[BaseDocument]

    tensor_columns: Dict[str, AbstractTensor]
    doc_columns: Dict[str, 'DocumentArrayStacked']
    da_columns: Dict[str, ListAdvanceIndex['DocumentArrayStacked']]
    any_columns: Dict[str, ListAdvanceIndex]

    def __init__(
        self,
        tensor_columns: Dict[str, AbstractTensor],
        doc_columns: Dict[str, 'DocumentArrayStacked'],
        da_columns: Dict[str, ListAdvanceIndex['DocumentArrayStacked']],
        any_columns: Dict[str, ListAdvanceIndex],
        document_type: Type[BaseDocument],
        tensor_type: Type[AbstractTensor] = NdArray,
    ):
        self.tensor_columns = tensor_columns
        self.doc_columns = doc_columns
        self.da_columns = da_columns
        self.any_columns = any_columns

        self.document_type = document_type
        self.tensor_type = tensor_type

        self.columns = ChainMap(
            self.tensor_columns,
            self.doc_columns,
            self.da_columns,
            self.any_columns,
        )

    @classmethod
    def from_docs(
        cls: Type[T],
        docs: Sequence[BaseDocument],
        document_type: Type[BaseDocument],
        tensor_type: Type[AbstractTensor],
    ) -> T:

        tensor_columns: Dict[str, AbstractTensor] = dict()
        doc_columns: Dict[str, 'DocumentArrayStacked'] = dict()
        da_columns: Dict[str, ListAdvanceIndex['DocumentArrayStacked']] = dict()
        any_columns: Dict[str, ListAdvanceIndex] = dict()

        docs = (
            docs
            if isinstance(docs, DocumentArray)
            else DocumentArray.__class_getitem__(document_type)(docs)
        )

        for field_name, field in document_type.__fields__.items():
            field_type = document_type._get_field_type(field_name)

            if is_tensor_union(field_type):
                field_type = tensor_type
            if isinstance(field_type, type):
                if tf_available and issubclass(field_type, TensorFlowTensor):
                    # tf.Tensor does not allow item assignment, therefore the optimized way
                    # of initializing an empty array and assigning values to it iteratively
                    # does not work here, therefore handle separately.
                    tf_stack = []
                    for i, doc in enumerate(docs):
                        val = getattr(doc, field_name)
                        if val is None:
                            val = tensor_type.get_comp_backend().none_value()
                        tf_stack.append(val.tensor)

                    stacked: tf.Tensor = tf.stack(tf_stack)
                    tensor_columns[field_name] = TensorFlowTensor(stacked)

                elif issubclass(field_type, AbstractTensor):

                    tensor = getattr(docs[0], field_name)
                    column_shape = (
                        (len(docs), *tensor.shape)
                        if tensor is not None
                        else (len(docs),)
                    )
                    tensor_columns[field_name] = field_type._docarray_from_native(
                        field_type.get_comp_backend().empty(
                            column_shape,
                            dtype=tensor.dtype if hasattr(tensor, 'dtype') else None,
                            device=tensor.device if hasattr(tensor, 'device') else None,
                        )
                    )

                    for i, doc in enumerate(docs):
                        val = getattr(doc, field_name)
                        if val is None:
                            val = tensor_type.get_comp_backend().none_value()

                        cast(AbstractTensor, tensor_columns[field_name])[i] = val

                elif issubclass(field_type, BaseDocument):
                    doc_columns[field_name] = getattr(docs, field_name).stack()

                elif issubclass(field_type, DocumentArray):
                    docs = list()
                    for doc in docs:
                        docs.append(getattr(doc, field_name).stack())
                    da_columns[field_name] = ListAdvanceIndex(docs)
                else:
                    any_columns[field_name] = ListAdvanceIndex(
                        getattr(docs, field_name)
                    )
            else:
                any_columns[field_name] = ListAdvanceIndex(getattr(docs, field_name))

        return cls(
            tensor_columns,
            doc_columns,
            da_columns,
            any_columns,
            document_type,
            tensor_type,
        )

    def __len__(self) -> int:
        return len(self.any_columns['id'])  # TODO what if ID are None ?

    def __getitem__(self: T, item: IndexIterType) -> T:
        if isinstance(item, tuple):
            item = list(item)
        tensor_columns = {key: col[item] for key, col in self.tensor_columns.items()}
        doc_columns = {key: col[item] for key, col in self.doc_columns.items()}
        da_columns = {key: col[item] for key, col in self.da_columns.items()}
        any_columns = {key: col[item] for key, col in self.any_columns.items()}

        return ColumnStorage(
            tensor_columns,
            doc_columns,
            da_columns,
            any_columns,
            self.document_type,
            self.tensor_type,
        )


class ColumnStorageView(dict, MutableMapping[str, Any]):
    index: int
    storage: ColumnStorage

    def __init__(self, index: int, storage: ColumnStorage):
        super().__init__()
        self.index = index
        self.storage = storage

    def __getitem__(self, name: str) -> Any:
        return self.storage.columns[name][self.index]

    def __setitem__(self, name, value) -> None:
        self.storage.columns[name][self.index] = value

    def __delitem__(self, key):
        raise RuntimeError('Cannot delete an item from a StorageView')

    def __iter__(self):
        return self.storage.columns.keys()

    def __len__(self):
        return len(self.storage.columns)
