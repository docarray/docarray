from collections import ChainMap
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    MutableSequence,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)

from pydantic import BaseConfig, parse_obj_as

from docarray.array.any_array import AnyDocArray
from docarray.array.doc_list.doc_list import DocList
from docarray.array.doc_vec.column_storage import ColumnStorage, ColumnStorageView
from docarray.array.doc_vec.list_advance_indexing import ListAdvancedIndexing
from docarray.base_doc import BaseDoc
from docarray.base_doc.mixins.io import _type_to_protobuf
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import is_tensor_union
from docarray.utils._internal.misc import is_tf_available, is_torch_available

if TYPE_CHECKING:
    from pydantic.fields import ModelField

    from docarray.proto import DocVecProto

torch_available = is_torch_available()
if torch_available:
    from docarray.typing import TorchTensor
else:
    TorchTensor = None  # type: ignore

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing import TensorFlowTensor  # noqa: F401
else:
    TensorFlowTensor = None  # type: ignore

T_doc = TypeVar('T_doc', bound=BaseDoc)
T = TypeVar('T', bound='DocVec')
IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]


class DocVec(AnyDocArray[T_doc]):
    """
    DocVec is a container of Documents appropriates to perform
    computation that require batches of data (ex: matrix multiplication, distance
    calculation, deep learning forward pass)

    A DocVec has a similar interface as [`DocList`][docarray.array.DocList]
    but with an underlying implementation that is column based instead of row based.
    Each field of the schema of the `DocVec` (the `.doc_type` which is a
    [`BaseDoc`][docarray.BaseDoc]) will be stored in a column.

    If the field is a tensor, the data from all Documents will be stored as a single
    doc_vec (torch/np/tf) tensor.

    If the tensor field is `AnyTensor` or a Union of tensor types, the
    `.tensor_type` will be used to determine the type of the doc_vec column.

    If the field is another [`BaseDoc`][docarray.BaseDoc] the column will be another
    `DocVec` that follows the schema of the nested Document.

    If the field is a [`DocList`][docarray.DocList] or `DocVec` then the column will
    be a list of `DocVec`.

    For any other type the column is a Python list.

    Every `Document` inside a `DocVec` is a view into the data columns stored at the
    `DocVec` level. The `BaseDoc` does not hold any data itself. The behavior of
    this Document "view" is similar to the behavior of `view = tensor[i]` in
    numpy/PyTorch.

    :param docs: a homogeneous sequence of `BaseDoc`
    :param tensor_type: Tensor Class used to wrap the doc_vec tensors. This is useful
        if the BaseDoc of this DocVec has some undefined tensor type like
        AnyTensor or Union of NdArray and TorchTensor
    """

    doc_type: Type[T_doc]

    def __init__(
        self: T,
        docs: Sequence[T_doc],
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):
        self.tensor_type = tensor_type

        tensor_columns: Dict[str, AbstractTensor] = dict()
        doc_columns: Dict[str, 'DocVec'] = dict()
        docs_vec_columns: Dict[str, ListAdvancedIndexing['DocVec']] = dict()
        any_columns: Dict[str, ListAdvancedIndexing] = dict()

        if len(docs) == 0:
            raise ValueError(f'docs {docs}: should not be empty')
        docs = (
            docs
            if isinstance(docs, DocList)
            else DocList.__class_getitem__(self.doc_type)(docs)
        )

        for field_name, field in self.doc_type.__fields__.items():
            # here we iterate over the field of the docs schema, and we collect the data
            # from each document and put them in the corresponding column
            field_type = self.doc_type._get_field_type(field_name)

            if is_tensor_union(field_type):
                field_type = tensor_type
            if isinstance(field_type, type):
                if tf_available and issubclass(field_type, TensorFlowTensor):
                    # tf.Tensor does not allow item assignment, therefore the
                    # optimized way
                    # of initializing an empty array and assigning values to it
                    # iteratively
                    # does not work here, therefore handle separately.
                    tf_stack = []
                    for i, doc in enumerate(docs):
                        val = getattr(doc, field_name)
                        if val is None:
                            val = TensorFlowTensor(
                                tensor_type.get_comp_backend().none_value()
                            )
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

                elif issubclass(field_type, BaseDoc):
                    doc_columns[field_name] = getattr(docs, field_name).to_doc_vec(
                        tensor_type=self.tensor_type
                    )

                elif issubclass(field_type, AnyDocArray):
                    docs_list = list()
                    for doc in docs:
                        docs_nested = getattr(doc, field_name)
                        if isinstance(docs_nested, DocList):
                            docs_nested = docs_nested.to_doc_vec(
                                tensor_type=self.tensor_type
                            )
                        docs_list.append(docs_nested)
                    docs_vec_columns[field_name] = ListAdvancedIndexing(docs_list)
                else:
                    any_columns[field_name] = ListAdvancedIndexing(
                        getattr(docs, field_name)
                    )
            else:
                any_columns[field_name] = ListAdvancedIndexing(
                    getattr(docs, field_name)
                )

        self._storage = ColumnStorage(
            tensor_columns,
            doc_columns,
            docs_vec_columns,
            any_columns,
            tensor_type,
        )

    @classmethod
    def from_columns_storage(cls: Type[T], storage: ColumnStorage) -> T:
        """
        Create a DocVec directly from a storage object
        :param storage: the underlying storage.
        :return: a DocVec
        """
        docs = cls.__new__(cls)
        docs.tensor_type = storage.tensor_type
        docs._storage = storage
        return docs

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[T_doc]],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, cls):
            return value
        elif isinstance(value, DocList.__class_getitem__(cls.doc_type)):
            return cast(T, value.to_doc_vec())
        elif isinstance(value, Sequence):
            return cls(value)
        elif isinstance(value, Iterable):
            return cls(list(value))
        else:
            raise TypeError(f'Expecting an Iterable of {cls.doc_type}')

    def to(self: T, device: str) -> T:
        """Move all tensors of this DocVec to the given device

        :param device: the device to move the data to
        """
        for field, col_tens in self._storage.tensor_columns.items():
            self._storage.tensor_columns[field] = col_tens.get_comp_backend().to_device(
                col_tens, device
            )

        for field, col_doc in self._storage.doc_columns.items():
            self._storage.doc_columns[field] = col_doc.to(device)
        for _, col_da in self._storage.docs_vec_columns.items():
            for docs in col_da:
                docs.to(device)

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

    def __getitem__(self: T, item: Union[int, IndexIterType]) -> Union[T_doc, T]:
        if item is None:
            return self  # PyTorch behaviour
        # multiple docs case
        if isinstance(item, (slice, Iterable)):
            return self.__class__.from_columns_storage(self._storage[item])
        # single doc case
        return self.doc_type.from_view(ColumnStorageView(item, self._storage))

    def _get_data_column(
        self: T,
        field: str,
    ) -> Union[MutableSequence, 'DocVec', AbstractTensor]:
        """Return one column of the data

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        if field in self._storage.any_columns.keys():
            return self._storage.any_columns[field].data
        elif field in self._storage.docs_vec_columns.keys():
            return self._storage.docs_vec_columns[field].data
        elif field in self._storage.columns.keys():
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

    @no_type_check
    def __setitem__(self: T, key, value):
        # single doc case
        if not isinstance(key, (slice, Iterable)):
            if not isinstance(value, self.doc_type):
                raise ValueError(f'{value} is not a {self.doc_type}')

            for field, value in value.dict().items():
                self._storage.columns[field][key] = value  # todo we might want to
                # define a safety mechanism in someone put a wrong value
        else:
            # multiple docs case
            self._set_data_and_columns(key, value)

    def _set_data_and_columns(
        self: T,
        index_item: Union[Tuple, Iterable, slice],
        value: Union[T, DocList[T_doc]],
    ) -> None:
        """Delegates the setting to the data and the columns.

        :param index_item: the key used as index. Needs to be a valid index for both
            DocList (data) and column types (torch/tensorflow/numpy tensors)
        :value: the value to set at the `key` location
        """
        if isinstance(index_item, tuple):
            index_item = list(index_item)

        # set data and prepare columns
        processed_value: T
        if isinstance(value, DocList):
            if not issubclass(value.doc_type, self.doc_type):
                raise TypeError(
                    f'{value} schema : {value.doc_type} is not compatible with '
                    f'this DocVec schema : {self.doc_type}'
                )
            processed_value = cast(
                T, value.to_doc_vec(tensor_type=self.tensor_type)
            )  # we need to copy data here

        elif isinstance(value, DocVec):
            if not issubclass(value.doc_type, self.doc_type):
                raise TypeError(
                    f'{value} schema : {value.doc_type} is not compatible with '
                    f'this DocVec schema : {self.doc_type}'
                )
            processed_value = value
        else:
            raise TypeError(f'Can not set a DocVec with {type(value)}')

        for field, col in self._storage.columns.items():
            col[index_item] = processed_value._storage.columns[field]

    def _set_data_column(
        self: T,
        field: str,
        values: Union[
            Sequence[DocList[T_doc]],
            Sequence[Any],
            T,
            DocList,
            AbstractTensor,
        ],
    ) -> None:
        """Set all Documents in this DocList using the passed values

        :param field: name of the fields to set
        :values: the values to set at the DocList level
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
            values_ = parse_obj_as(
                DocVec.__class_getitem__(self._storage.doc_columns[field].doc_type),
                values,
            )
            self._storage.doc_columns[field] = values_

        elif field in self._storage.docs_vec_columns.keys():
            values_ = cast(Sequence[DocList[T_doc]], values)
            # TODO here we should actually check if this is correct
            self._storage.docs_vec_columns[field] = values_
        elif field in self._storage.any_columns.keys():
            # TODO here we should actually check if this is correct
            values_ = cast(Sequence, values)
            self._storage.any_columns[field] = values_
        else:
            raise KeyError(f'{field} is not a valid field for this DocList')

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
    def from_protobuf(cls: Type[T], pb_msg: 'DocVecProto') -> T:
        """create a Document from a protobuf message"""
        storage = ColumnStorage(
            pb_msg.tensor_columns,
            pb_msg.doc_columns,
            pb_msg.docs_vec_columns,
            pb_msg.any_columns,
        )

        return cls.from_columns_storage(storage)

    def to_protobuf(self) -> 'DocVecProto':
        """Convert DocList into a Protobuf message"""
        from docarray.proto import (
            DocListProto,
            DocVecProto,
            ListOfAnyProto,
            ListOfDocArrayProto,
            NdArrayProto,
        )

        da_proto = DocListProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        doc_columns_proto: Dict[str, DocVecProto] = dict()
        tensor_columns_proto: Dict[str, NdArrayProto] = dict()
        da_columns_proto: Dict[str, ListOfDocArrayProto] = dict()
        any_columns_proto: Dict[str, ListOfAnyProto] = dict()

        for field, col_doc in self._storage.doc_columns.items():
            doc_columns_proto[field] = col_doc.to_protobuf()
        for field, col_tens in self._storage.tensor_columns.items():
            tensor_columns_proto[field] = col_tens.to_protobuf()
        for field, col_da in self._storage.docs_vec_columns.items():
            list_proto = ListOfDocArrayProto()
            for docs in col_da:
                list_proto.data.append(docs.to_protobuf())
            da_columns_proto[field] = list_proto
        for field, col_any in self._storage.any_columns.items():
            list_proto = ListOfAnyProto()
            for data in col_any:
                list_proto.data.append(_type_to_protobuf(data))
            any_columns_proto[field] = list_proto

        return DocVecProto(
            doc_columns=doc_columns_proto,
            tensor_columns=tensor_columns_proto,
            docs_vec_columns=da_columns_proto,
            any_columns=any_columns_proto,
        )

    def to_doc_list(self: T) -> DocList[T_doc]:
        """Convert DocVec into a DocList.

        Note this destroys the arguments and returns a new DocList
        """

        unstacked_doc_column: Dict[str, DocList] = dict()
        unstacked_da_column: Dict[str, List[DocList]] = dict()
        unstacked_tensor_column: Dict[str, List[AbstractTensor]] = dict()
        unstacked_any_column = self._storage.any_columns

        for field, doc_col in self._storage.doc_columns.items():
            unstacked_doc_column[field] = doc_col.to_doc_list()

        for field, da_col in self._storage.docs_vec_columns.items():
            unstacked_da_column[field] = [docs.to_doc_list() for docs in da_col]

        for field, tensor_col in list(self._storage.tensor_columns.items()):
            # list is needed here otherwise we cannot delete the column
            unstacked_tensor_column[field] = list()
            for tensor in tensor_col:
                tensor_copy = tensor.get_comp_backend().copy(tensor)
                unstacked_tensor_column[field].append(tensor_copy)

            del self._storage.tensor_columns[field]

        unstacked_column = ChainMap(  # type: ignore
            unstacked_any_column,  # type: ignore
            unstacked_tensor_column,  # type: ignore
            unstacked_da_column,  # type: ignore
            unstacked_doc_column,  # type: ignore
        )  # type: ignore

        docs = []

        for i in range(len(self)):
            data = {field: col[i] for field, col in unstacked_column.items()}
            docs.append(self.doc_type.construct(**data))

        del self._storage

        return DocList.__class_getitem__(self.doc_type).construct(docs)

    def traverse_flat(
        self,
        access_path: str,
    ) -> Union[List[Any], 'TorchTensor', 'NdArray']:
        nodes = list(AnyDocArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocArray._flatten_one_level(nodes)

        cls_to_check = (NdArray, TorchTensor) if TorchTensor is not None else (NdArray,)

        if len(flattened) == 1 and isinstance(flattened[0], cls_to_check):
            return flattened[0]
        else:
            return flattened
