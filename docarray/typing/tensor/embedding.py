from typing import TypeVar

from docarray.proto import NodeProto
from docarray.typing.tensor import Tensor

T = TypeVar('T', bound='Embedding')


class Embedding(Tensor):
    def _to_node_protobuf(self: T, field: str = 'tensor') -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """

        return super()._to_node_protobuf(field='embedding')
