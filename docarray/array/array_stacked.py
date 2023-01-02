from collections import defaultdict
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Type,
    TypeVar,
    Union,
)

from typing_inspect import is_union_type

from docarray.array.abstract_array import AnyDocumentArray
from docarray.array.array import DocumentArray
from docarray.document import AnyDocument, BaseDocument
from docarray.typing import AnyTensor, NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import DocumentArrayStackedProto
    from docarray.typing import TorchTensor
    from docarray.typing.tensor.abstract_tensor import AbstractTensor

try:
    import torch
except ImportError:
    torch_imported = False
else:
    from docarray.typing import TorchTensor

    torch_imported = True

T = TypeVar('T', bound='DocumentArrayStacked')


class DocumentArrayStacked(AnyDocumentArray):
    """
    DocumentArrayStacked is a container of Documents appropriates to perform
    computation that require batches of data (ex: matrix multiplication, distance
    calculation, deep learning forward pass)

    A DocumentArrayStacked is similar to {class}`~docarray.array.DocumentArray`
    but the field of the Document that are {class}`~docarray.typing.AnyTensor` are
    stacked into a batches of AnyTensor. Like {class}`~docarray.array.DocumentArray`
    you can be precise a Document schema by using the `DocumentArray[MyDocument]`
    syntax where MyDocument is a Document class  (i.e. schema).
    This creates a DocumentArray that can only contains Documents of
    the type 'MyDocument'.

    :param docs: a DocumentArray

    """

    document_type: Type[BaseDocument] = AnyDocument

    def __init__(
        self: T,
        docs: DocumentArray,
    ):
        self._docs = docs
        self.tensor_type = self._docs.tensor_type
        self._columns: Dict[
            str, Union['TorchTensor', T, NdArray]
        ] = self._create_columns(docs, tensor_type=self.tensor_type)

    def from_document_array(self: T, docs: DocumentArray):
        self._docs = docs
        self._columns = self._create_columns(docs, tensor_type=self.tensor_type)
        self.tensor_type = self._docs.tensor_type

    @classmethod
    def _create_columns(
        cls: Type[T], docs: DocumentArray, tensor_type: Type['AbstractTensor']
    ) -> Dict[str, Union['TorchTensor', T, NdArray]]:

        columns_fields = list()
        for field_name, field in cls.document_type.__fields__.items():
            field_type = field.type_
            if is_union_type(field_type):
                if field.type_ == AnyTensor:
                    columns_fields.append(field_name)
            else:
                is_torch_subclass = (
                    issubclass(field_type, torch.Tensor) if torch_imported else False
                )

                if (
                    is_torch_subclass
                    or issubclass(field_type, BaseDocument)
                    or issubclass(field_type, NdArray)
                ):
                    columns_fields.append(field_name)

        columns: Dict[str, Union['TorchTensor', T, NdArray]] = dict()

        columns_to_stack: DefaultDict[
            str, Union[List['TorchTensor'], List[NdArray], List[BaseDocument]]
        ] = defaultdict(  # type: ignore
            list  # type: ignore
        )  # type: ignore

        for doc in docs:
            for field_to_stack in columns_fields:
                columns_to_stack[field_to_stack].append(getattr(doc, field_to_stack))
                setattr(doc, field_to_stack, None)

        for field_to_stack, to_stack in columns_to_stack.items():

            type_ = cls.document_type.__fields__[field_to_stack].type_
            if isinstance(type_, type):
                if issubclass(type_, BaseDocument):
                    columns[field_to_stack] = DocumentArray.__class_getitem__(type_)(
                        to_stack
                    ).stack()

                elif issubclass(type_, (NdArray, TorchTensor)):
                    columns[field_to_stack] = type_.__docarray_stack__(to_stack)  # type: ignore # noqa: E501

            elif type_ == AnyTensor:  # should be properly handled with union
                columns[field_to_stack] = tensor_type.__docarray_stack__(to_stack)  # type: ignore # noqa: E501

        return columns

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        if field in self._columns.keys():
            return self._columns[field]
        else:
            return getattr(self._docs, field)

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, 'TorchTensor', 'NdArray'],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        if field in self._columns.keys() and not isinstance(values, List):
            self._columns[field] = values
        else:
            setattr(self._docs, field, values)

    def __getitem__(self, item):  # note this should handle slices
        doc = self._docs[item]
        # NOTE: this could be speed up by using a cache
        for field in self._columns.keys():
            setattr(doc, field, self._columns[field][item])
        return doc

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem_without_columns__(self, item):  # note this should handle slices
        """Return the document at the given index with the columns item put to None"""
        doc = self._docs[item]
        for field in self._columns.keys():
            setattr(doc, field, None)
        return doc

    def __iter_without_columns__(self):
        for i in range(len(self)):
            yield self.__getitem_without_columns__(i)

    def __len__(self):
        return len(self._docs)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayStackedProto') -> T:
        """create a Document from a protobuf message"""

        docs = DocumentArray(
            cls.document_type.from_protobuf(doc_proto)
            for doc_proto in pb_msg.list_.docs
        )
        da = cls(DocumentArray([]))

        da._docs = docs
        da._columns = pb_msg.columns
        return da

    def to_protobuf(self) -> 'DocumentArrayStackedProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import (
            DocumentArrayProto,
            DocumentArrayStackedProto,
            UnionArrayProto,
        )

        da_proto = DocumentArrayProto()
        for doc in self.__iter_without_columns__():
            da_proto.docs.append(doc.to_protobuf())

        columns_proto: Dict[str, UnionArrayProto] = dict()
        for field, column in self._columns.items():
            if isinstance(column, DocumentArrayStacked):
                columns_proto[field] = UnionArrayProto(
                    document_array=DocumentArrayProto(stack=column.to_protobuf())
                )
            elif isinstance(column, AbstractTensor):
                columns_proto[field] = UnionArrayProto(ndarray=column.to_protobuf())

        return DocumentArrayStackedProto(list_=da_proto, columns=columns_proto)

    def unstack(self: T) -> DocumentArray:
        """Convert DocumentArrayStacked into a DocumentArray.

        Note this destroys the arguments and returns a new DocumentArray
        """
        for i, doc in enumerate(self._docs):
            for field in self._columns.keys():
                val = self._columns[field]
                setattr(doc, field, val[i])

                # NOTE: here we might need to copy the tensor
                # see here
                # https://discuss.pytorch.org/t/what-happened-to-a-view-of-a-tensor
                # -when-the-original-tensor-is-deleted/167294 # noqa: E501

        for field in list(self._columns.keys()):
            # list needed here otherwise we are modifying the dict while iterating
            del self._columns[field]

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
            self.from_document_array(da_unstacked)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, Iterable[BaseDocument]],
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

        if len(flattened) == 1 and isinstance(flattened[0], (NdArray, TorchTensor)):
            return flattened[0]
        else:
            return flattened
