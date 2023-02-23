from contextlib import contextmanager
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from pydantic import parse_obj_as

from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array.array import DocumentArray
from docarray.base_document import AnyDocument, BaseDocument
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._typing import is_tensor_union
from docarray.utils.misc import is_tf_available, is_torch_available

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import DocumentArrayStackedProto

torch_available = is_torch_available()
if torch_available:
    from docarray.typing import TorchTensor
else:
    TorchTensor = None  # type: ignore

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore

    from docarray.typing import TensorFlowTensor
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
    _docs: DocumentArray

    def __init__(
        self: T,
        docs: Optional[Union[DocumentArray, Iterable[T_doc]]] = None,
        tensor_type: Type['AbstractTensor'] = NdArray,
    ):
        self._doc_columns: Dict[str, 'DocumentArrayStacked'] = {}
        self._tensor_columns: Dict[str, AbstractTensor] = {}
        self.tensor_type = tensor_type

        self.from_iterable_document(docs)

    def from_iterable_document(
        self: T, docs: Optional[Union[DocumentArray, Iterable[T_doc]]]
    ):
        self._docs = (
            docs
            if isinstance(docs, DocumentArray)
            else DocumentArray.__class_getitem__(self.document_type)(
                docs, tensor_type=self.tensor_type
            )
        )
        self.tensor_type = self._docs.tensor_type
        self._doc_columns, self._tensor_columns = self._create_columns(
            self._docs, tensor_type=self.tensor_type
        )

    @classmethod
    def _from_da_and_columns(
        cls: Type[T],
        docs: DocumentArray,
        doc_columns: Dict[str, 'DocumentArrayStacked'],
        tensor_columns: Dict[str, AbstractTensor],
    ) -> T:
        """Create a DocumentArrayStacked from a DocumentArray
        and an associated dict of columns"""
        # below __class_getitem__ is called explicitly instead
        # of doing DocumentArrayStacked[docs.document_type]
        # because mypy has issues with class[...] notation at runtime.
        # see bug here: https://github.com/python/mypy/issues/13026
        # as of 2023-01-05 it should be fixed on mypy master, though, see
        # here: https://github.com/python/typeshed/issues/4819#issuecomment-1354506442

        da_stacked: T = DocumentArray.__class_getitem__(cls.document_type)([]).stack()
        da_stacked._doc_columns = doc_columns
        da_stacked._tensor_columns = tensor_columns
        da_stacked._docs = docs
        return da_stacked

    def to(self: T, device: str) -> T:
        """Move all tensors of this DocumentArrayStacked to the given device

        :param device: the device to move the data to
        """
        for field in self._tensor_columns.keys():
            col_tens: AbstractTensor = self._tensor_columns[field]
            self._tensor_columns[field] = col_tens.__class__._docarray_from_native(
                col_tens.get_comp_backend().to_device(col_tens, device)
            )
        for field in self._doc_columns.keys():
            col_doc: 'DocumentArrayStacked' = self._doc_columns[field]
            col_doc.to(device)
        return self

    @classmethod
    def _create_columns(
        cls: Type[T], docs: DocumentArray, tensor_type: Type[AbstractTensor]
    ) -> Tuple[Dict[str, 'DocumentArrayStacked'], Dict[str, AbstractTensor]]:

        if len(docs) == 0:
            return {}, {}

        doc_columns: Dict[str, DocumentArrayStacked] = dict()
        tensor_columns: Dict[str, AbstractTensor] = dict()

        for field_name, field in cls.document_type.__fields__.items():
            field_type = field.outer_type_

            if is_tensor_union(field_type):
                field_type = tensor_type

            if tf_available and isinstance(
                getattr(docs[0], field_name), TensorFlowTensor
            ):
                # tf.Tensor does not allow item assignment, therefore the optimized way
                # of initializing an empty array and assigning values to it iteratively
                # does not work here, therefore handle separately.
                tf_stack = []
                for i, doc in enumerate(docs):
                    val = getattr(doc, field_name)
                    if val is None:
                        val = tensor_type.get_comp_backend().none_value()
                    tf_stack.append(val.tensor)
                    del val.tensor

                stacked: tf.Tensor = tf.stack(tf_stack)
                tensor_columns[field_name] = TensorFlowTensor(stacked)
                for i, doc in enumerate(docs):
                    val = getattr(doc, field_name)
                    x = tensor_columns[field_name][i].tensor
                    val.tensor = x

            elif isinstance(field_type, type):
                if issubclass(field_type, AbstractTensor):
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

                        # If the stacked tensor is rank 1, the individual tensors are
                        # rank 0 (scalar)
                        # This is problematic because indexing a rank 1 tensor in numpy
                        # returns a value instead of a tensor
                        # We thus chose to convert the individual rank 0 tensors to rank 1
                        # This does mean that stacking rank 0 tensors will transform them
                        # to rank 1
                        tensor = tensor_columns[field_name]
                        if tensor.get_comp_backend().n_dim(tensor) == 1:
                            setattr(
                                doc, field_name, tensor_columns[field_name][i : i + 1]
                            )
                        else:
                            setattr(doc, field_name, tensor_columns[field_name][i])
                        del val

                elif issubclass(field_type, BaseDocument):
                    doc_columns[field_name] = getattr(docs, field_name).stack()
                    for i, doc in enumerate(docs):
                        setattr(doc, field_name, doc_columns[field_name][i])

                elif issubclass(field_type, DocumentArray):
                    for doc in docs:
                        setattr(doc, field_name, getattr(doc, field_name).stack())

        return doc_columns, tensor_columns

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, 'DocumentArrayStacked', AbstractTensor]:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        if field in self._doc_columns.keys():
            return self._doc_columns[field]
        elif field in self._tensor_columns.keys():
            return self._tensor_columns[field]
        else:
            return getattr(self._docs, field)

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, AbstractTensor],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        if field in self._doc_columns.keys() and not isinstance(values, List):
            values_ = cast(T, values)
            self._doc_columns[field] = values_
        elif field in self._tensor_columns.keys() and not isinstance(values, List):

            type_ = cast(AbstractTensor, self.document_type._get_field_type(field))
            shaped_type = (
                type_.__unparametrizedcls__
                if type_.__unparametrizedcls__ is not None
                else type_
            )
            values__ = parse_obj_as(shaped_type, values)  # type: ignore
            # TODO: here we should validate that the shape is correct, maybe create a
            # __unparametrizedcls__[len(self), X ,Y] ...
            self._tensor_columns[field] = values__
            for i, doc in enumerate(self):
                setattr(doc, field, self._tensor_columns[field][i])
        else:
            setattr(self._docs, field, values)

    @overload
    def __getitem__(self: T, item: int) -> T_doc:
        ...

    @overload
    def __getitem__(self: T, item: IndexIterType) -> T:
        ...

    def __getitem__(self, item):
        if item is None:
            return self  # PyTorch behaviour
        # multiple docs case
        if isinstance(item, (slice, Iterable)):
            item_ = cast(Iterable, item)
            return self._get_from_data_and_columns(item_)
        # single doc case
        doc = self._docs[item]
        return doc

    @overload
    def __setitem__(self: T, key: int, value: T_doc):
        ...

    @overload
    def __setitem__(self: T, key: IndexIterType, value: T):
        ...

    def __setitem__(self: T, key: Union[int, IndexIterType], value: Union[T, T_doc]):
        # multiple docs case
        if isinstance(key, (slice, Iterable)):
            self._set_data_and_columns(key, value)
        else:
            # single doc case
            self._docs[key] = value
            for field in chain(self._tensor_columns.keys(), self._doc_columns.keys()):
                self._tensor_columns[field][key] = getattr(value, field)

    @overload
    def __delitem__(self: T, key: int) -> None:
        ...

    @overload
    def __delitem__(self: T, key: IndexIterType) -> None:
        ...

    def __delitem__(self, key) -> None:
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement '
            f'__del_item__. You are trying to delete an element'
            f'from {self.__class__.__name__} which is not '
            f'designed for this operation. Please `unstack`'
            f' before doing the deletion'
        )

    def _get_from_data_and_columns(self: T, item: Union[Tuple, Iterable]) -> T:
        """Delegates the access to the data and the columns,
        and combines into a stacked da.

        :param item: the item used as index. Needs to be a valid index for both
            DocumentArray (data) and column types (torch/tensorflow/numpy tensors)
        :return: a DocumentArrayStacked, indexed according to `item`
        """
        if isinstance(item, tuple):
            item = list(item)
        # get documents
        docs_indexed = self._docs[item]
        # get doc columns
        doc_columns_indexed = {k: col[item] for k, col in self._doc_columns.items()}
        doc_columns_indexed_ = cast(
            Dict[str, 'DocumentArrayStacked'], doc_columns_indexed
        )
        # get tensor columns
        tensor_columns_indexed = {
            k: col[item] for k, col in self._tensor_columns.items()
        }
        return self._from_da_and_columns(
            docs_indexed, doc_columns_indexed_, tensor_columns_indexed
        )

    def _set_data_and_columns(
        self: T,
        index_item: Union[Tuple, Iterable, slice],
        value: Union[T, BaseDocument],
    ):
        """Delegates the setting to the data and the columns.

        :param index_item: the key used as index. Needs to be a valid index for both
            DocumentArray (data) and column types (torch/tensorflow/numpy tensors)
        :value: the value to set at the `key` location
        """
        if isinstance(index_item, tuple):
            index_item = list(index_item)

        # set data and prepare columns
        doc_cols_to_set: Dict[str, DocumentArrayStacked]
        tens_cols_to_set: Dict[str, AbstractTensor]
        if isinstance(value, DocumentArray):
            self._docs[index_item] = value
            doc_cols_to_set, tens_cols_to_set = self._create_columns(
                value, self.tensor_type
            )
        elif isinstance(value, BaseDocument):
            self._docs[index_item] = value
            doc_cols_to_set, tens_cols_to_set = self._create_columns(
                DocumentArray.__class_getitem__(self.document_type)([value]),
                self.tensor_type,
            )
        elif isinstance(value, DocumentArrayStacked):
            self._docs[index_item] = value._docs
            doc_cols_to_set = value._doc_columns
            tens_cols_to_set = value._tensor_columns
        else:
            raise TypeError(f'Can not set a DocumentArrayStacked with {type(value)}')

        # set columns
        for col_key in self._doc_columns.keys():
            self._doc_columns[col_key][index_item] = doc_cols_to_set[col_key]
        for col_key in self._tensor_columns.keys():
            self._tensor_columns[col_key][index_item] = tens_cols_to_set[col_key]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self._docs)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayStackedProto') -> T:
        """create a Document from a protobuf message"""

        docs: DocumentArray = DocumentArray(
            cls.document_type.from_protobuf(doc_proto)
            for doc_proto in pb_msg.list_.docs
        )
        da: T = cls(DocumentArray([]))

        da._docs = docs
        da._doc_columns = pb_msg.doc_columns
        da._tensor_columns = pb_msg.tensor_columns
        return da

    def to_protobuf(self) -> 'DocumentArrayStackedProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import (
            DocumentArrayProto,
            DocumentArrayStackedProto,
            NdArrayProto,
        )

        da_proto = DocumentArrayProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        doc_columns_proto: Dict[str, DocumentArrayStackedProto] = dict()
        tens_columns_proto: Dict[str, NdArrayProto] = dict()
        for field, col_doc in self._doc_columns.items():
            doc_columns_proto[field] = col_doc.to_protobuf()
        for field, col_tens in self._tensor_columns.items():
            tens_columns_proto[field] = col_tens.to_protobuf()

        return DocumentArrayStackedProto(
            list_=da_proto,
            doc_columns=doc_columns_proto,
            tensor_columns=tens_columns_proto,
        )

    def unstack(self: T) -> DocumentArray:
        """Convert DocumentArrayStacked into a DocumentArray.

        Note this destroys the arguments and returns a new DocumentArray
        """
        for i, doc in enumerate(self._docs):
            for field in self._doc_columns.keys():
                val_doc = self._doc_columns[field]
                setattr(doc, field, val_doc[i])

            for field in self._tensor_columns.keys():
                val_tens = self._tensor_columns[field]
                setattr(doc, field, val_tens[i])

                # NOTE: here we might need to copy the tensor
                # see here
                # https://discuss.pytorch.org/t/what-happened-to-a-view-of-a-tensor
                # -when-the-original-tensor-is-deleted/167294 # noqa: E501

        for field_name, _ in self._docs.document_type.__fields__.items():
            field_type = self.document_type._get_field_type(field_name)
            if isinstance(field_type, type) and issubclass(field_type, DocumentArray):
                for doc in self._docs:
                    setattr(doc, field_name, getattr(doc, field_name).unstack())

        for field in list(self._doc_columns.keys()):
            # list needed here otherwise we are modifying the dict while iterating
            del self._doc_columns[field]

        for field in list(self._tensor_columns.keys()):
            # list needed here otherwise we are modifying the dict while iterating
            del self._tensor_columns[field]

        da_list = self._docs
        return da_list

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
        try:
            da_unstacked = DocumentArrayStacked.unstack(self)
            yield da_unstacked
        finally:
            self._docs = da_unstacked
            self.from_iterable_document(da_unstacked)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[T_doc]],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, cls):
            return value
        elif isinstance(value, Iterable):
            return cls(DocumentArray(value))
        else:
            raise TypeError(f'Expecting an Iterable of {cls.document_type}')

    def traverse_flat(
        self: 'AnyDocumentArray',
        access_path: str,
    ) -> Union[List[Any], 'TorchTensor', 'NdArray']:
        nodes = list(AnyDocumentArray._traverse(node=self, access_path=access_path))
        flattened = AnyDocumentArray._flatten_one_level(nodes)

        cls_to_check = (NdArray, TorchTensor) if TorchTensor is not None else (NdArray,)

        if len(flattened) == 1 and isinstance(flattened[0], cls_to_check):
            return flattened[0]
        else:
            return flattened
