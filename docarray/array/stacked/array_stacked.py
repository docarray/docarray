from collections import ChainMap
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from pydantic import BaseConfig, parse_obj_as

from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array.array import DocumentArray
from docarray.array.stacked.column_storage import ColumnStorage, ColumnStorageView
from docarray.base_document import AnyDocument, BaseDocument
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils.misc import is_tf_available, is_torch_available

if TYPE_CHECKING:
    from pydantic.fields import ModelField

    from docarray.proto import DocumentArrayStackedProto

torch_available = is_torch_available()
if torch_available:
    from docarray.typing import TorchTensor
else:
    TorchTensor = None  # type: ignore

tf_available = is_tf_available()
if tf_available:
    from docarray.typing import TensorFlowTensor  # noqa: F401
else:
    TensorFlowTensor = None  # type: ignore

T_doc = TypeVar('T_doc', bound=BaseDocument)
T = TypeVar('T', bound='DocumentArrayStacked')
IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


class DocumentArrayStacked(AnyDocumentArray[T_doc]):
    """
    DocumentArrayStacked is a container of Documents appropriates to perform
    computation that require batches of data (ex: matrix multiplication, distance
    calculation, deep learning forward pass)

    A DocumentArrayStacked is similar to {class}`~docarray.array.DocumentArray`
    but the field of the Document that are {class}`~docarray.typing.AnyTensor` are
    stacked into a batches of AnyTensor. Like {class}`~docarray.array.DocumentArray`
    you can be precise a Document schema by using the `DocumentArray[MyDocument]`
    syntax where MyDocument is a Document class (i.e. schema).
    This creates a DocumentArray that can only contains Documents of
    the type 'MyDocument'.

    :param docs: a DocumentArray
    :param tensor_type: Class used to wrap the stacked tensors

    """

    document_type: Type[BaseDocument] = AnyDocument
    _storage: ColumnStorage

    def __init__(
        self: T,
        docs: Sequence[T_doc],
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):
        self.tensor_type = tensor_type
        self._storage = ColumnStorage.from_docs(
            docs, document_type=self.document_type, tensor_type=self.tensor_type
        )

    @classmethod
    def from_columns_storage(cls: Type[T], storage: ColumnStorage) -> T:
        """
        Create a DocumentArrayStack directly from a storage object
        :param storage: the underlying storage.
        :return: a DocumentArrayStack
        """
        da = cls.__new__(cls)
        da.tensor_type = storage.tensor_type
        da._storage = storage
        return da

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[T_doc]],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, cls):
            return value
        elif isinstance(value, DocumentArray[cls.document_type]):
            return value.stack()
        elif isinstance(value, Iterable):
            return cls(value)
        else:
            raise TypeError(f'Expecting an Iterable of {cls.document_type}')

    def to(self: T, device: str) -> T:
        """Move all tensors of this DocumentArrayStacked to the given device

        :param device: the device to move the data to
        """
        for field, col_tens in self._storage.tensor_columns.items():
            self._storage.tensor_columns[field] = col_tens.get_comp_backend().to_device(
                col_tens, device
            )

        for field, col_doc in self._storage.doc_columns.items():
            self._storage.doc_columns[field] = col_doc.to(device)
        for _, col_da in self._storage.da_columns.items():
            for da in col_da:
                da.to(device)

        return self

    ################################################
    # Accessing data : Indexing / Getitem related  #
    ################################################

    @overload
    def __getitem__(self: T, item: int) -> T_doc:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    def __getitem__(self, item: Union[int, IndexIterType]) -> Union[T_doc, T]:
        if item is None:
            return self  # PyTorch behaviour
        # multiple docs case
        if isinstance(item, (slice, Iterable)):
            return self.__class__.from_columns_storage(self._storage[item])
        # single doc case
        doc = self.document_type.from_view(ColumnStorageView(item, self._storage))
        return doc

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, 'DocumentArrayStacked', AbstractTensor]:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        if field in self._storage.columns.keys():
            return self._storage.columns[field]
        else:
            raise ValueError(f'{field} does not exist in {self}')

    ####################################
    # Updating data : Setitem related  #
    ####################################

    @overload
    def __setitem__(self: T, key: int, value: T_doc):
        ...

    @overload
    def __setitem__(self: T, key: IndexIterType, value: T):
        ...

    def __setitem__(self: T, key: Union[int, IndexIterType], value: Union[T, T_doc]):
        # single doc case
        if not isinstance(key, (slice, Iterable)):
            key_ = cast(int, key)
            doc = cast(T_doc, value)
            if not isinstance(doc, self.document_type):
                raise ValueError(f'{doc} is not a {self.document_type}')

            for field, value in doc.dict().items():
                self._storage.columns[field][key_] = value  # todo we might want to
                # define a safety mechanism in someone put a wrong value

        else:
            # multiple docs case
            self._set_data_and_columns(key, value)

    def _set_data_and_columns(
        self: T,
        index_item: Union[Tuple, Iterable, slice],
        value: Union[T, DocumentArray[T_doc]],
    ) -> None:
        """Delegates the setting to the data and the columns.

        :param index_item: the key used as index. Needs to be a valid index for both
            DocumentArray (data) and column types (torch/tensorflow/numpy tensors)
        :value: the value to set at the `key` location
        """
        if isinstance(index_item, tuple):
            index_item = list(index_item)

        # set data and prepare columns
        processed_value: T
        if isinstance(value, DocumentArray):
            if not issubclass(value.document_type, self.document_type):
                raise TypeError(
                    f'{value} schema : {value.document_type} is not compatible with '
                    f'this DocumentArrayStacked schema : {self.document_type}'
                )
            processed_value = value.stack()  # we need to copy data here

        elif isinstance(value, DocumentArrayStacked):
            if not issubclass(value.document_type, self.document_type):
                raise TypeError(
                    f'{value} schema : {value.document_type} is not compatible with '
                    f'this DocumentArrayStacked schema : {self.document_type}'
                )
            processed_value = value
        else:
            raise TypeError(f'Can not set a DocumentArrayStacked with {type(value)}')

        for field, col in self._storage.columns.items():
            col[index_item] = processed_value._storage.columns[field]
            # TODO complex indexing will not work here

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, DocumentArray, AbstractTensor],
    ) -> None:
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """

        if len(values) != len(self._storage):
            raise ValueError(
                f'{values} has not the right length, expected '
                f'{len(self._storage)} , got {len(values)}'
            )
        if field in self._storage.tensor_columns.keys():
            validation_class = (
                self._storage.tensor_columns[field].__unparametrizedcls__
                or self._storage.tensor_columns[field].__class__
            )
            # TODO shape check should be handle by the tensor validation

            values = parse_obj_as(validation_class, values)
            self._storage.tensor_columns[field] = values

        elif field in self._storage.doc_columns.keys():

            values = parse_obj_as(
                DocumentArrayStacked[self._storage.doc_columns[field].document_type],
                values,
            )

            self._storage.doc_columns[field] = values

        elif field in self._storage.da_columns.keys():
            self._storage.da_columns[field] = values
        elif field in self._storage.any_columns.keys():
            self._storage.any_columns[field] = values
        else:
            raise KeyError(f'{field} is not a valid field for this DocumentArray')

    ####################
    # Deleting data    #
    ####################

    def __delitem__(self, key: Union[int, IndexIterType]) -> None:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement '
            f'__del_item__. You are trying to delete an element'
            f'from {self.__class__.__name__} which is not '
            f'designed for this operation. Please `unstack`'
            f' before doing the deletion'
        )

    ####################
    # Sequence related #
    ####################
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._storage)

    ####################
    # IO related       #
    ####################

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayStackedProto') -> T:
        """create a Document from a protobuf message"""

        raise NotImplementedError

    def to_protobuf(self) -> 'DocumentArrayStackedProto':
        """Convert DocumentArray into a Protobuf message"""
        raise NotImplementedError

    def unstack(self: T) -> DocumentArray[T_doc]:
        """Convert DocumentArrayStacked into a DocumentArray.

        Note this destroys the arguments and returns a new DocumentArray
        """

        unstacked_doc_column: Dict[str, DocumentArray] = dict()
        unstacked_da_column: Dict[str, List[DocumentArray]] = dict()
        unstacked_tensor_column: Dict[str, List[AbstractTensor]] = dict()
        unstacked_any_column = self._storage.any_columns

        for field, doc_col in self._storage.doc_columns.items():
            unstacked_doc_column[field] = doc_col.unstack()

        for field, da_col in self._storage.da_columns.items():
            unstacked_da_column[field] = [da.unstack() for da in da_col]

        for field, tensor_col in self._storage.tensor_columns.items():
            unstacked_tensor_column[field] = list()
            for tensor in tensor_col:
                tensor_copy = tensor.get_comp_backend().copy(tensor)
                unstacked_tensor_column[field].append(tensor_copy)

            # del self._storage.tensor_columns[field]
            # todo fix this should be uncommented

        unstacked_column = ChainMap(
            unstacked_any_column,
            unstacked_tensor_column,
            unstacked_da_column,
            unstacked_doc_column,
        )

        docs = []

        for i in range(len(self)):

            data = {field: col[i] for field, col in unstacked_column.items()}
            docs.append(self.document_type.construct(**data))

        return DocumentArray[self.document_type].construct(
            docs, tensor_type=self.tensor_type
        )

    @contextmanager
    def unstacked_mode(self):
        """
        Context manager to put the DocumentArrayStacked in unstacked mode and stack it
        when exiting the context manager.
        EXAMPLE USAGE
        .. code-block:: python
            with da.unstacked_mode():
                ...
        """
        raise NotImplementedError

    def traverse_flat(
        self,
        access_path: str,
    ) -> Union[List[Any], 'TorchTensor', 'NdArray']:
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten_one_level(nodes)

        cls_to_check = (NdArray, TorchTensor) if TorchTensor is not None else (NdArray,)

        if len(flattened) == 1 and isinstance(flattened[0], cls_to_check):
            return flattened[0]
        else:
            return flattened
