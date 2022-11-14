from pydantic import AnyUrl as BaseAnyUrl

from docarray.document.base_node import BaseNode
from docarray.proto import NodeProto


class AnyUrl(BaseAnyUrl, BaseNode):
    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(text=str(self))
