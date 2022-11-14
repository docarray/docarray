from typing import Any, Dict

from pydantic.tools import parse_obj_as

from docarray.document.abstract_document import AbstractDocument
from docarray.document.base_node import BaseNode
from docarray.proto import DocumentProto, NodeProto
from docarray.typing import AnyUrl, Embedding, ImageUrl, Tensor


class ProtoMixin(AbstractDocument, BaseNode):
    @classmethod
    def from_protobuf(cls, pb_msg: 'DocumentProto') -> 'ProtoMixin':
        """create a Document from a protobuf message"""
        from docarray import DocumentArray

        fields: Dict[str, Any] = {}

        for field in pb_msg.data:
            value = pb_msg.data[field]

            content_type = value.WhichOneof('content')

            # this if else statement need to be refactored it is too long
            # the check should be delegated to the type level
            if content_type == 'tensor':
                fields[field] = Tensor._read_from_proto(value.tensor)
            elif content_type == 'embedding':
                fields[field] = Embedding._read_from_proto(value.embedding)
            elif content_type == 'any_url':
                fields[field] = parse_obj_as(AnyUrl, value.any_url)
            elif content_type == 'image_url':
                fields[field] = parse_obj_as(ImageUrl, value.image_url)
            elif content_type == 'text':
                fields[field] = value.text
            elif content_type == 'nested':
                fields[field] = cls._get_nested_document_class(field).from_protobuf(
                    value.nested
                )  # we get to the parent class
            elif content_type == 'chunks':

                fields[field] = DocumentArray.from_protobuf(
                    value.chunks
                )  # we get to the parent class
            elif content_type is None:
                fields[field] = None
            else:
                raise ValueError(
                    f'type {content_type} is not supported for deserialization'
                )

        return cls(**fields)

    def to_protobuf(self) -> 'DocumentProto':
        """Convert Document into a Protobuf message.

        :return: the protobuf message
        """
        data = {}
        for field, value in self:
            try:
                if isinstance(value, BaseNode):
                    nested_item = value._to_node_protobuf()

                elif type(value) is str:
                    nested_item = NodeProto(text=value)

                elif type(value) is bytes:
                    nested_item = NodeProto(blob=value)
                elif value is None:
                    nested_item = NodeProto()
                else:
                    raise ValueError(f'field {field} with {value} is not supported')

                data[field] = nested_item

            except RecursionError as ex:
                if len(ex.args) >= 1:
                    ex.args = (
                        (
                            f'Field `{field}` contains cyclic reference in memory. '
                            'Could it be your Document is referring to itself?'
                        ),
                    )
                raise
            except Exception as ex:
                if len(ex.args) >= 1:
                    ex.args = (f'Field `{field}` is problematic',) + ex.args
                raise

        return DocumentProto(data=data)

    def _to_node_protobuf(self) -> NodeProto:
        """Convert Document into a NodeProto protobuf message. This function should be
        called when the Document is nest into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(nested=self.to_protobuf())
