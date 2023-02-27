from collections import ChainMap
from typing import TYPE_CHECKING, Any, Dict, List, MutableMapping, Sequence, Type, cast

from docarray.array.array.array import DocumentArray
from docarray.base_document import BaseDocument
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


class Storage:

    document_type: Type[BaseDocument]

    tensor_storage: Dict[str, AbstractTensor]
    document_storage: Dict[str, 'DocumentArrayStacked']
    da_storage: Dict[str, List['DocumentArrayStacked']]
    any_storage: Dict[str, List]

    def __init__(
        self,
        docs: Sequence[BaseDocument],
        document_type: Type[BaseDocument],
        tensor_type: Type[AbstractTensor],
    ):

        self.document_type = document_type
        self.tensor_type = tensor_type

        self.tensor_storage = dict()
        self.document_storage = dict()
        self.da_storage = dict()
        self.any_storage = dict()

        docs = (
            docs
            if isinstance(docs, DocumentArray)
            else DocumentArray.__class_getitem__(self.document_type)(docs)
        )

        for field_name, field in self.document_type.__fields__.items():
            field_type = self.document_type._get_field_type(field_name)

            if is_tensor_union(field_type):
                field_type = tensor_type

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
                self.tensor_storage[field_name] = TensorFlowTensor(stacked)

            elif issubclass(field_type, AbstractTensor):

                tensor = getattr(docs[0], field_name)
                column_shape = (
                    (len(docs), *tensor.shape) if tensor is not None else (len(docs),)
                )
                self.tensor_storage[field_name] = field_type._docarray_from_native(
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

                    cast(AbstractTensor, self.tensor_storage[field_name])[i] = val

            elif issubclass(field_type, BaseDocument):
                self.document_storage[field_name] = getattr(docs, field_name).stack()

            elif issubclass(field_type, DocumentArray):
                self.da_storage[field_name] = list()
                for doc in docs:
                    self.da_storage[field_name].append(getattr(doc, field_name).stack())
            else:
                self.any_storage[field_name] = getattr(docs, field_name)

        self.columns = ChainMap(
            self.tensor_storage,
            self.document_storage,
            self.da_storage,
            self.any_storage,
        )


class StorageView(MutableMapping[str, Any]):
    index: int
    storage: Storage

    def __init__(self, index: int, storage: Storage):
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
