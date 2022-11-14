from abc import ABC, abstractmethod

from docarray.proto import NodeProto


class BaseNode(ABC):
    """
    A DocumentNode is an object than can be nested inside a Document.
    A Document itself is a DocumentNode as well as prebuilt type
    """

    @abstractmethod
    def _to_nested_item_protobuf(self) -> 'NodeProto':
        """Convert itself into a nested item protobuf message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        ...
