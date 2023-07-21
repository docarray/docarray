from collections import ChainMap
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
    overload,
)

import numpy as np
import orjson
from pydantic import BaseConfig, parse_obj_as
from typing_inspect import typingGenericAlias

from docarray.array.any_array import AnyDocArray
from docarray.array.doc_list.doc_list import DocList
from docarray.array.doc_list.io import IOMixinArray
from docarray.array.doc_vec.column_storage import ColumnStorage, ColumnStorageView
from docarray.array.list_advance_indexing import ListAdvancedIndexing
from docarray.base_doc import AnyDoc, BaseDoc
from docarray.base_doc.mixins.io import _type_to_protobuf
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import is_tensor_union, safe_issubclass
from docarray.utils._internal.misc import (
    is_jax_available,
    is_tf_available,
    is_torch_available,
)

if TYPE_CHECKING:
    import csv

    from pydantic.fields import ModelField

    from docarray.proto import (
        DocVecProto,
        ListOfDocArrayProto,
        ListOfDocVecProto,
        NdArrayProto,
    )

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

jnp_available = is_jax_available()
if jnp_available:
    import jax.numpy as jnp  # type: ignore

    from docarray.typing import JaxArray  # noqa: F401
else:
    JaxArray = None  # type: ignore

T_doc = TypeVar('T_doc', bound=BaseDoc)
T = TypeVar('T', bound='DocVec')
T_io_mixin = TypeVar('T_io_mixin', bound='IOMixinArray')

IndexIterType = Union[slice, Iterable[int], Iterable[bool], None]

NONE_NDARRAY_PROTO_SHAPE = (0,)
NONE_NDARRAY_PROTO_DTYPE = 'None'


def _none_ndarray_proto() -> 'NdArrayProto':
    from docarray.proto import NdArrayProto

    zeros_arr = parse_obj_as(NdArray, np.zeros(NONE_NDARRAY_PROTO_SHAPE))
    nd_proto = NdArrayProto()
    nd_proto.dense.buffer = zeros_arr.tobytes()
    nd_proto.dense.ClearField('shape')
    nd_proto.dense.shape.extend(list(zeros_arr.shape))
    nd_proto.dense.dtype = NONE_NDARRAY_PROTO_DTYPE

    return nd_proto


def _none_docvec_proto() -> 'DocVecProto':
    from docarray.proto import DocVecProto

    return DocVecProto()


def _none_list_of_docvec_proto() -> 'ListOfDocArrayProto':
    from docarray.proto import ListOfDocVecProto

    return ListOfDocVecProto()


def _is_none_ndarray_proto(proto: 'NdArrayProto') -> bool:
    return (
        proto.dense.shape == list(NONE_NDARRAY_PROTO_SHAPE)
        and proto.dense.dtype == NONE_NDARRAY_PROTO_DTYPE
    )


def _is_none_docvec_proto(proto: 'DocVecProto') -> bool:
    return (
        proto.tensor_columns == {}
        and proto.doc_columns == {}
        and proto.docs_vec_columns == {}
        and proto.any_columns == {}
    )


def _is_none_list_of_docvec_proto(proto: 'ListOfDocVecProto') -> bool:
    from docarray.proto import ListOfDocVecProto

    return isinstance(proto, ListOfDocVecProto) and len(proto.data) == 0


class DocVec(IOMixinArray, AnyDocArray[T_doc]):
    """
    DocVec is a container of Documents appropriates to perform
    computation that require batches of data (ex: matrix multiplication, distance
    calculation, deep learning forward pass)

    A DocVec has a similar interface as [`DocList`][docarray.array.DocList]
    but with an underlying implementation that is column based instead of row based.
    Each field of the schema of the `DocVec` (the `.doc_type` which is a
    [`BaseDoc`][docarray.BaseDoc]) will be stored in a column.

    If the field is a tensor, the data from all Documents will be stored as a single
    (torch/np/tf) tensor.

    If the tensor field is `AnyTensor` or a Union of tensor types, the
    `.tensor_type` will be used to determine the type of the column.

    If the field is another [`BaseDoc`][docarray.BaseDoc] the column will be another
    `DocVec` that follows the schema of the nested Document.

    If the field is a [`DocList`][docarray.DocList] or `DocVec` then the column will
    be a list of `DocVec`.

    For any other type the column is a Python list.

    Every `Document` inside a `DocVec` is a view into the data columns stored at the
    `DocVec` level. The `BaseDoc` does not hold any data itself. The behavior of
    this Document "view" is similar to the behavior of `view = tensor[i]` in
    numpy/PyTorch.

    !!! note
        DocVec supports optional fields. Nevertheless if a field is optional it needs to
        be homogeneous. This means that if the first document has a None value all of the
        other documents should have a None value as well.
    !!! note
        If one field is Optional the column will be stored
        * as None if the first doc is as the field as None
        * as a normal column otherwise that cannot contain None value

    :param docs: a homogeneous sequence of `BaseDoc`
    :param tensor_type: Tensor Class used to wrap the doc_vec tensors. This is useful
        if the BaseDoc of this DocVec has some undefined tensor type like
        AnyTensor or Union of NdArray and TorchTensor
    """

    doc_type: Type[T_doc] = BaseDoc  # type: ignore

    def __init__(
        self: T,
        docs: Sequence[T_doc],
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):

        if (
            not hasattr(self, 'doc_type')
            or self.doc_type == AnyDoc
            or self.doc_type == BaseDoc
        ):
            raise TypeError(
                f'{self.__class__.__name__} does not precise a doc_type. You probably should do'
                f'docs = DocVec[MyDoc](docs) instead of DocVec(docs)'
            )
        self.tensor_type = tensor_type
        self._is_unusable = False

        tensor_columns: Dict[str, Optional[AbstractTensor]] = dict()
        doc_columns: Dict[str, Optional['DocVec']] = dict()
        docs_vec_columns: Dict[str, Optional[ListAdvancedIndexing['DocVec']]] = dict()
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
            field_type: Type = self.doc_type._get_field_type(field_name)

            is_field_required = self.doc_type.__fields__[field_name].required

            first_doc_is_none = getattr(docs[0], field_name) is None

            def _verify_optional_field_of_docs(docs):

                if is_field_required:
                    if first_doc_is_none:
                        raise ValueError(
                            f'Field {field_name} is None for {docs[0]} even though it is required'
                        )

                if first_doc_is_none:
                    for i, doc in enumerate(docs):
                        if getattr(doc, field_name) is not None:
                            raise ValueError(
                                f'Field {field_name} is put to None for the first doc. This mean that '
                                f'all of the other docs should have this field set to None as well. '
                                f'This is not the case for {doc} at index {i}'
                            )

            def _check_doc_field_not_none(field_name, doc):
                if getattr(doc, field_name) is None:
                    raise ValueError(
                        f'Field {field_name} is None for {doc} even though it is not None for the first doc'
                    )

            if is_tensor_union(field_type):
                field_type = tensor_type
            # all generic tensor types such as AnyTensor, ImageTensor, etc. are subclasses of AbstractTensor.
            # Perform check only if the field_type is not an alias and is a subclass of AbstractTensor
            elif not isinstance(field_type, typingGenericAlias) and safe_issubclass(
                field_type, AbstractTensor
            ):
                # check if the tensor associated with the field_name in the document is a subclass of the tensor_type
                # e.g. if the field_type is AnyTensor but the type(docs[0][field_name]) is ImageTensor,
                # then we change the field_type to ImageTensor, since AnyTensor is a union of all the tensor types
                # and does not override any methods of specific tensor types
                tensor = getattr(docs[0], field_name)
                if safe_issubclass(tensor.__class__, tensor_type):
                    field_type = tensor_type

            if isinstance(field_type, type):
                if tf_available and safe_issubclass(field_type, TensorFlowTensor):
                    # tf.Tensor does not allow item assignment, therefore the
                    # optimized way
                    # of initializing an empty array and assigning values to it
                    # iteratively
                    # does not work here, therefore handle separately.

                    if first_doc_is_none:
                        _verify_optional_field_of_docs(docs)
                        tensor_columns[field_name] = None
                    else:
                        tf_stack = []
                        for i, doc in enumerate(docs):
                            val = getattr(doc, field_name)
                            _check_doc_field_not_none(field_name, doc)
                            tf_stack.append(val.tensor)

                        stacked: tf.Tensor = tf.stack(tf_stack)
                        tensor_columns[field_name] = TensorFlowTensor(stacked)
                elif jnp_available and issubclass(field_type, JaxArray):
                    if first_doc_is_none:
                        _verify_optional_field_of_docs(docs)
                        tensor_columns[field_name] = None
                    else:
                        tf_stack = []
                        for i, doc in enumerate(docs):
                            val = getattr(doc, field_name)
                            _check_doc_field_not_none(field_name, doc)
                            tf_stack.append(val.tensor)

                        jax_stacked: jnp.ndarray = jnp.stack(tf_stack)
                        tensor_columns[field_name] = JaxArray(jax_stacked)

                elif safe_issubclass(field_type, AbstractTensor):
                    if first_doc_is_none:
                        _verify_optional_field_of_docs(docs)
                        tensor_columns[field_name] = None
                    else:
                        tensor = getattr(docs[0], field_name)
                        column_shape = (
                            (len(docs), *tensor.shape)
                            if tensor is not None
                            else (len(docs),)
                        )
                        tensor_columns[field_name] = field_type._docarray_from_native(
                            field_type.get_comp_backend().empty(
                                column_shape,
                                dtype=tensor.dtype
                                if hasattr(tensor, 'dtype')
                                else None,
                                device=tensor.device
                                if hasattr(tensor, 'device')
                                else None,
                            )
                        )

                        for i, doc in enumerate(docs):
                            _check_doc_field_not_none(field_name, doc)
                            val = getattr(doc, field_name)
                            cast(AbstractTensor, tensor_columns[field_name])[i] = val

                elif safe_issubclass(field_type, BaseDoc):
                    if first_doc_is_none:
                        _verify_optional_field_of_docs(docs)
                        doc_columns[field_name] = None
                    else:
                        if is_field_required:
                            doc_columns[field_name] = getattr(
                                docs, field_name
                            ).to_doc_vec(tensor_type=self.tensor_type)
                        else:
                            doc_columns[field_name] = DocList.__class_getitem__(
                                field_type
                            )(getattr(docs, field_name)).to_doc_vec(
                                tensor_type=self.tensor_type
                            )

                elif safe_issubclass(field_type, AnyDocArray):
                    if first_doc_is_none:
                        _verify_optional_field_of_docs(docs)
                        docs_vec_columns[field_name] = None
                    else:
                        docs_list = list()
                        for doc in docs:
                            docs_nested = getattr(doc, field_name)
                            _check_doc_field_not_none(field_name, doc)
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
        elif isinstance(value, DocList):
            if (
                safe_issubclass(value.doc_type, cls.doc_type)
                or value.doc_type == cls.doc_type
            ):
                return cast(T, value.to_doc_vec())
            else:
                raise ValueError(f'DocVec[value.doc_type] is not compatible with {cls}')
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
            if col_tens is not None:
                self._storage.tensor_columns[
                    field
                ] = col_tens.get_comp_backend().to_device(col_tens, device)

        for field, col_doc in self._storage.doc_columns.items():
            if col_doc is not None:
                self._storage.doc_columns[field] = col_doc.to(device)
        for _, col_da in self._storage.docs_vec_columns.items():
            if col_da is not None:
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
    ) -> Union[MutableSequence, 'DocVec', AbstractTensor, None]:
        """Return one column of the data

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        if field in self._storage.any_columns.keys():
            return self._storage.any_columns[field]
        elif field in self._storage.docs_vec_columns.keys():
            return self._storage.docs_vec_columns[field]
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
            if not safe_issubclass(value.doc_type, self.doc_type):
                raise TypeError(
                    f'{value} schema : {value.doc_type} is not compatible with '
                    f'this DocVec schema : {self.doc_type}'
                )
            processed_value = cast(
                T, value.to_doc_vec(tensor_type=self.tensor_type)
            )  # we need to copy data here

        elif isinstance(value, DocVec):
            if not safe_issubclass(value.doc_type, self.doc_type):
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
            None,
        ],
    ) -> None:
        """Set all Documents in this DocList using the passed values

        :param field: name of the fields to set
        :values: the values to set at the DocList level
        """
        if values is None:
            if field in self._storage.tensor_columns.keys():
                self._storage.tensor_columns[field] = values
            elif field in self._storage.doc_columns.keys():
                self._storage.doc_columns[field] = values
            elif field in self._storage.docs_vec_columns.keys():
                self._storage.docs_vec_columns[field] = values
            elif field in self._storage.any_columns.keys():
                raise ValueError(
                    f'column {field} cannot be set to None, try to pass '
                    f'a list of None instead'
                )
            else:
                raise ValueError(f'{field} does not exist in {self}')

        else:
            if len(values) != len(self._storage):
                raise ValueError(
                    f'{values} has not the right length, expected '
                    f'{len(self._storage)} , got {len(values)}'
                )
            if field in self._storage.tensor_columns.keys():

                col = self._storage.tensor_columns[field]
                if col is not None:
                    validation_class = col.__unparametrizedcls__ or col.__class__
                else:
                    validation_class = self.doc_type.__fields__[field].type_

                # TODO shape check should be handle by the tensor validation

                values = parse_obj_as(validation_class, values)
                self._storage.tensor_columns[field] = values

            elif field in self._storage.doc_columns.keys():
                values_ = parse_obj_as(
                    DocVec.__class_getitem__(self.doc_type._get_field_type(field)),
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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DocVec):
            return False
        if self.doc_type != other.doc_type:
            return False
        if self.tensor_type != other.tensor_type:
            return False
        if self._storage != other._storage:
            return False
        return True

    ####################
    # IO related       #
    ####################

    @classmethod
    def _get_proto_class(cls: Type[T]):
        from docarray.proto import DocVecProto

        return DocVecProto

    def _docarray_to_json_compatible(self) -> Dict[str, Dict[str, Any]]:
        tup = self._storage.columns_json_compatible()
        return tup._asdict()

    @classmethod
    def from_json(
        cls: Type[T],
        file: Union[str, bytes, bytearray],
        tensor_type: Type[AbstractTensor] = NdArray,
    ) -> T:
        """Deserialize JSON strings or bytes into a `DocList`.

        :param file: JSON object from where to deserialize a `DocList`
        :param tensor_type: the tensor type to use for the tensor columns.
            Could be NdArray, TorchTensor, or TensorFlowTensor. Defaults to NdArray.
            All tensors of the output DocVec will be of this type.
        :return: the deserialized `DocList`
        """
        json_columns = orjson.loads(file)
        return cls._from_json_col_dict(json_columns, tensor_type=tensor_type)

    @classmethod
    def _from_json_col_dict(
        cls: Type[T],
        json_columns: Dict[str, Any],
        tensor_type: Type[AbstractTensor] = NdArray,
    ) -> T:

        tensor_cols = json_columns['tensor_columns']
        doc_cols = json_columns['doc_columns']
        docs_vec_cols = json_columns['docs_vec_columns']
        any_cols = json_columns['any_columns']

        for key, col in tensor_cols.items():
            if col is not None:
                tensor_cols[key] = parse_obj_as(tensor_type, col)
            else:
                tensor_cols[key] = None

        for key, col in doc_cols.items():
            if col is not None:
                col_doc_type = cls.doc_type._get_field_type(key)
                doc_cols[key] = DocVec.__class_getitem__(
                    col_doc_type
                )._from_json_col_dict(col, tensor_type=tensor_type)
            else:
                doc_cols[key] = None

        for key, col in docs_vec_cols.items():
            if col is not None:
                col_doc_type = cls.doc_type._get_field_type(key).doc_type
                col_ = ListAdvancedIndexing(
                    DocVec.__class_getitem__(col_doc_type)._from_json_col_dict(
                        vec, tensor_type=tensor_type
                    )
                    for vec in col
                )
                docs_vec_cols[key] = col_
            else:
                docs_vec_cols[key] = None

        for key, col in any_cols.items():
            if col is not None:
                col_type = cls.doc_type._get_field_type(key)
                col_type = (
                    col_type
                    if cls.doc_type.__fields__[key].required
                    else Optional[col_type]
                )
                col_ = ListAdvancedIndexing(parse_obj_as(col_type, val) for val in col)
                any_cols[key] = col_
            else:
                any_cols[key] = None

        return cls.from_columns_storage(
            ColumnStorage(
                tensor_cols, doc_cols, docs_vec_cols, any_cols, tensor_type=tensor_type
            )
        )

    @classmethod
    def from_protobuf(
        cls: Type[T], pb_msg: 'DocVecProto', tensor_type: Type[AbstractTensor] = NdArray
    ) -> T:
        """create a DocVec from a protobuf message
        :param pb_msg: the protobuf message to deserialize
        :param tensor_type: the tensor type to use for the tensor columns.
            Could be NdArray, TorchTensor, or TensorFlowTensor. Defaults to NdArray.
            All tensors of the output DocVec will be of this type.
        :return: The deserialized DocVec
        """

        tensor_columns: Dict[str, Optional[AbstractTensor]] = {}
        doc_columns: Dict[str, Optional['DocVec']] = {}
        docs_vec_columns: Dict[str, Optional[ListAdvancedIndexing['DocVec']]] = {}
        any_columns: Dict[str, ListAdvancedIndexing] = {}

        for tens_col_name, tens_col_proto in pb_msg.tensor_columns.items():
            if _is_none_ndarray_proto(tens_col_proto):
                # handle values that were None before serialization
                tensor_columns[tens_col_name] = None
            else:
                tensor_columns[tens_col_name] = tensor_type.from_protobuf(
                    tens_col_proto
                )

        for doc_col_name, doc_col_proto in pb_msg.doc_columns.items():
            if _is_none_docvec_proto(doc_col_proto):
                # handle values that were None before serialization
                doc_columns[doc_col_name] = None
            else:
                col_doc_type: Type = cls.doc_type._get_field_type(doc_col_name)
                doc_columns[doc_col_name] = DocVec.__class_getitem__(
                    col_doc_type
                ).from_protobuf(doc_col_proto, tensor_type=tensor_type)

        for docs_vec_col_name, docs_vec_col_proto in pb_msg.docs_vec_columns.items():
            vec_list: Optional[ListAdvancedIndexing]
            if _is_none_list_of_docvec_proto(docs_vec_col_proto):
                # handle values that were None before serialization
                vec_list = None
            else:
                vec_list = ListAdvancedIndexing()
                for doc_list_proto in docs_vec_col_proto.data:
                    col_doc_type = cls.doc_type._get_field_type(
                        docs_vec_col_name
                    ).doc_type
                    vec_list.append(
                        DocVec.__class_getitem__(col_doc_type).from_protobuf(
                            doc_list_proto, tensor_type=tensor_type
                        )
                    )
            docs_vec_columns[docs_vec_col_name] = vec_list

        for any_col_name, any_col_proto in pb_msg.any_columns.items():
            any_column: ListAdvancedIndexing = ListAdvancedIndexing()
            for node_proto in any_col_proto.data:
                content = cls.doc_type._get_content_from_node_proto(
                    node_proto, any_col_name
                )
                any_column.append(content)
            any_columns[any_col_name] = any_column

        storage = ColumnStorage(
            tensor_columns=tensor_columns,
            doc_columns=doc_columns,
            docs_vec_columns=docs_vec_columns,
            any_columns=any_columns,
            tensor_type=tensor_type,
        )

        return cls.from_columns_storage(storage)

    def to_protobuf(self) -> 'DocVecProto':
        """Convert DocVec into a Protobuf message"""
        from docarray.proto import (
            DocVecProto,
            ListOfAnyProto,
            ListOfDocArrayProto,
            ListOfDocVecProto,
            NdArrayProto,
        )

        doc_columns_proto: Dict[str, DocVecProto] = dict()
        tensor_columns_proto: Dict[str, NdArrayProto] = dict()
        da_columns_proto: Dict[str, ListOfDocArrayProto] = dict()
        any_columns_proto: Dict[str, ListOfAnyProto] = dict()

        for field, col_doc in self._storage.doc_columns.items():
            if col_doc is None:
                # put dummy empty DocVecProto for serialization
                doc_columns_proto[field] = _none_docvec_proto()
            else:
                doc_columns_proto[field] = col_doc.to_protobuf()
        for field, col_tens in self._storage.tensor_columns.items():
            if col_tens is None:
                # put dummy empty NdArrayProto for serialization
                tensor_columns_proto[field] = _none_ndarray_proto()
            else:
                tensor_columns_proto[field] = (
                    col_tens.to_protobuf() if col_tens is not None else None
                )
        for field, col_da in self._storage.docs_vec_columns.items():
            list_proto = ListOfDocVecProto()
            if col_da:
                for docs in col_da:
                    list_proto.data.append(docs.to_protobuf())
            else:
                # put dummy empty ListOfDocVecProto for serialization
                list_proto = _none_list_of_docvec_proto()
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

        unstacked_doc_column: Dict[str, Optional[DocList]] = dict()
        unstacked_da_column: Dict[str, Optional[List[DocList]]] = dict()
        unstacked_tensor_column: Dict[str, Optional[List[AbstractTensor]]] = dict()
        unstacked_any_column = self._storage.any_columns

        for field, doc_col in self._storage.doc_columns.items():
            unstacked_doc_column[field] = doc_col.to_doc_list() if doc_col else None

        for field, da_col in self._storage.docs_vec_columns.items():
            unstacked_da_column[field] = (
                [docs.to_doc_list() for docs in da_col] if da_col else None
            )

        for field, tensor_col in list(self._storage.tensor_columns.items()):
            # list is needed here otherwise we cannot delete the column
            if tensor_col is not None:
                tensors = list()
                for tensor in tensor_col:
                    tensor_copy = tensor.get_comp_backend().copy(tensor)
                    tensors.append(tensor_copy)

                unstacked_tensor_column[field] = tensors
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

        doc_type = self.doc_type

        # Setting _is_unusable will raise an Exception if someone interacts with this instance from hereon out.
        # I don't like relying on this state, but we can't override the getattr/setattr directly:
        # https://stackoverflow.com/questions/10376604/overriding-special-methods-on-an-instance
        self._is_unusable = True

        return DocList.__class_getitem__(doc_type).construct(docs)

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

    def to_csv(
        self, file_path: str, dialect: Union[str, 'csv.Dialect'] = 'excel'
    ) -> None:
        """
        DocVec does not support `.to_csv()`. This is because CSV is a row-based format
        while DocVec has a column-based data layout.
        To overcome this, do: `doc_vec.to_doc_list().to_csv(...)`.
        """
        raise NotImplementedError(
            f'{type(self)} does not support `.to_csv()`. This is because CSV is a row-based format'
            f'while {type(self)} has a column-based data layout. '
            f'To overcome this, do: `doc_vec.to_doc_list().to_csv(...)`.'
        )

    @classmethod
    def from_csv(
        cls: Type['T'],
        file_path: str,
        encoding: str = 'utf-8',
        dialect: Union[str, 'csv.Dialect'] = 'excel',
    ) -> 'T':
        """
        DocVec does not support `.from_csv()`. This is because CSV is a row-based format
        while DocVec has a column-based data layout.
        To overcome this, do: `DocList[MyDoc].from_csv(...).to_doc_vec()`.
        """
        raise NotImplementedError(
            f'{cls} does not support `.from_csv()`. This is because CSV is a row-based format while'
            f'{cls} has a column-based data layout. '
            f'To overcome this, do: `DocList[MyDoc].from_csv(...).to_doc_vec()`.'
        )
