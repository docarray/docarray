from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Type, TypeVar, Union

from docarray.array import AbstractDocumentArray, DocumentArray
from docarray.document import BaseDocument
from docarray.typing import NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.proto import DocumentArrayProto
    from docarray.typing import TorchTensor

try:
    import torch
except ImportError:
    torch_imported = False
else:
    from docarray.typing import TorchTensor

    torch_imported = True

T = TypeVar('T', bound='DocumentArrayStacked')


class DocumentArrayStacked(AbstractDocumentArray):
    def __init__(self: T, docs: DocumentArray):
        self._docs = docs

        self._columns: Dict[
            str, Union['TorchTensor', T, NdArray]
        ] = self._create_columns(docs)

    @classmethod
    def _create_columns(
        cls: Type[T], docs: DocumentArray
    ) -> Dict[str, Union['TorchTensor', T, NdArray]]:

        columns_fields = list()
        for field_name, field in cls.document_type.__fields__.items():

            is_torch_subclass = (
                issubclass(field.type_, torch.Tensor) if torch_imported else False
            )

            if (
                is_torch_subclass
                or issubclass(field.type_, BaseDocument)
                or issubclass(field.type_, NdArray)
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
            if issubclass(type_, BaseDocument):
                columns[field_to_stack] = DocumentArrayStacked.__class_getitem__(type_)(
                    to_stack
                )
            else:
                columns[field_to_stack] = type_.__docarray_stack__(to_stack)

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
        doc = super().__getitem__(item)
        # NOTE: this could be speed up by using a cache
        for field in self._columns.keys():
            setattr(doc, field, self._columns[field][item])
        return doc

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem_without_columns__(self, item):  # note this should handle slices
        """Return the document at the given index with the columns item put to None"""
        doc = super().__getitem__(item)
        for field in self._columns.keys():
            setattr(doc, field, None)
        return doc

    def __iter_without_columns__(self):
        for i in range(len(self)):
            yield self.__getitem_without_columns__(i)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message"""
        da = cls(
            DocumentArray(
                cls.document_type.from_protobuf(doc_proto)
                for doc_proto in pb_msg.stack.list_.docs
            )
        )
        da._columns = pb_msg.stack.columns
        return da

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import (
            DocumentArrayListProto,
            DocumentArrayProto,
            DocumentArrayStackedProto,
            UnionArrayProto,
        )

        da_proto = DocumentArrayListProto()
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
