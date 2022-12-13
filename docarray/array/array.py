from typing import TYPE_CHECKING, Iterable, List, Type, TypeVar, Union

from docarray.array.abstract_array import AbstractDocumentArray
from docarray.document import BaseDocument

if TYPE_CHECKING:
    from docarray.proto import DocumentArrayProto
    from docarray.typing import NdArray, TorchTensor


T = TypeVar('T', bound='DocumentArray')


class DocumentArray(AbstractDocumentArray, list[BaseDocument]):
    def __init__(self, docs: Iterable[BaseDocument]):
        super().__init__(doc_ for doc_ in docs)

    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        field_type = self.__class__.document_type._get_nested_document_class(field)

        if issubclass(field_type, BaseDocument):
            # calling __class_getitem__ ourselves is a hack otherwise mypy complain
            # most likely a bug in mypy though
            # bug reported here https://github.com/python/mypy/issues/14111
            return self.__class__.__class_getitem__(field_type)(
                (getattr(doc, field) for doc in self)
            )
        else:
            return [getattr(doc, field) for doc in self]

    def _set_array_attribute(
        self: T,
        field: str,
        values: Union[List, T, 'TorchTensor', 'NdArray'],
    ):
        """Set all Documents in this DocumentArray using the passed values

        :param field: name of the fields to extract
        :values: the values to set at the DocumentArray level
        """
        ...

        for doc, value in zip(self, values):
            setattr(doc, field, value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message"""
        return cls(
            cls.document_type.from_protobuf(doc_proto)
            for doc_proto in pb_msg.list_.docs
        )

    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        from docarray.proto import DocumentArrayListProto, DocumentArrayProto

        da_proto = DocumentArrayListProto()
        for doc in self:
            da_proto.docs.append(doc.to_protobuf())

        return DocumentArrayProto(list_=da_proto)
