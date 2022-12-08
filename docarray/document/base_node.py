from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docarray.proto import NodeProto


class BaseNode(ABC):
    """
    A DocumentNode is an object than can be nested inside a Document.
    A Document itself is a DocumentNode as well as prebuilt type
    """

    @abstractmethod
    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert itself into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        ...
