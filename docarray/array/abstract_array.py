from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Sequence, Type, TypeVar, Union

from docarray.document import BaseDocument, BaseNode

if TYPE_CHECKING:
    from docarray.proto import DocumentArrayProto, NodeProto
    from docarray.typing import NdArray, TorchTensor


T = TypeVar('T', bound='AbstractDocumentArray')


class AbstractDocumentArray(Sequence[BaseDocument], BaseNode):
    document_type: Type[BaseDocument]

    @abstractmethod
    def __init__(self, docs: Iterable[BaseDocument]):
        ...

    def __class_getitem__(cls, item: Type[BaseDocument]):
        if not issubclass(item, BaseDocument):
            raise ValueError(
                f'{cls.__name__}[item] item should be a Document not a {item} '
            )

        class _DocumentArrayTyped(cls):  # type: ignore
            document_type: Type[BaseDocument] = item

        for field in _DocumentArrayTyped.document_type.__fields__.keys():

            def _property_generator(val: str):
                def _getter(self):
                    return self._get_array_attribute(val)

                def _setter(self, value):
                    self._set_array_attribute(val, value)

                # need docstring for the property
                return property(fget=_getter, fset=_setter)

            setattr(_DocumentArrayTyped, field, _property_generator(field))
            # this generates property on the fly based on the schema of the item

        _DocumentArrayTyped.__name__ = f'DocumentArray[{item.__name__}]'
        _DocumentArrayTyped.__qualname__ = f'DocumentArray[{item.__name__}]'

        return _DocumentArrayTyped

    @abstractmethod
    def _get_array_attribute(
        self: T,
        field: str,
    ) -> Union[List, T, 'TorchTensor', 'NdArray']:
        """Return all values of the fields from all docs this array contains

        :param field: name of the fields to extract
        :return: Returns a list of the field value for each document
        in the array like container
        """
        ...

    @abstractmethod
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

    @classmethod
    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentArrayProto') -> T:
        """create a Document from a protobuf message"""
        ...

    @abstractmethod
    def to_protobuf(self) -> 'DocumentArrayProto':
        """Convert DocumentArray into a Protobuf message"""
        ...

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert a DocumentArray into a NodeProto protobuf message.
         This function should be called when a DocumentArray
        is nested into another Document that need to be converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(chunks=self.to_protobuf())
