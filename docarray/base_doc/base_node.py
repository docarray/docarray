from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Optional, Type

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T')


class BaseNode(ABC):
    """
    A DocumentNode is an object than can be nested inside a Document.
    A Document itself is a DocumentNode as well as prebuilt type
    """

    _proto_type_name: Optional[str] = None

    @abstractmethod
    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert itself into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        ...

    @classmethod
    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        ...

    def _docarray_to_json_compatible(self):
        """
        Convert itself into a json compatible object
        """
        ...
