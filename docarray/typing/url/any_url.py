from typing import Type, TypeVar

from pydantic import AnyUrl as BaseAnyUrl
from pydantic import parse_obj_as

from docarray.document.base_node import BaseNode
from docarray.proto import NodeProto

T = TypeVar('T', bound='AnyUrl')


class AnyUrl(BaseAnyUrl, BaseNode):
    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to
        be converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(any_url=str(self))

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
        """
        read url from a proto msg
        :param pb_msg:
        :return: url
        """
        return parse_obj_as(cls, pb_msg)
