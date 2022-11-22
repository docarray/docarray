from typing import Any, Dict, Type, TypeVar

from docarray.document.abstract_document import AbstractDocument
from docarray.document.base_node import BaseNode
from docarray.proto import DocumentProto, NodeProto
from docarray.typing import (
    ID,
    AnyUrl,
    Embedding,
    ImageUrl,
    Tensor,
    TextUrl,
    TorchTensor,
)

T = TypeVar('T', bound='ProtoMixin')


class ProtoMixin(AbstractDocument, BaseNode):
    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentProto') -> T:
        """create a Document from a protobuf message"""

        fields: Dict[str, Any] = {}

        for field in pb_msg.data:
            value = pb_msg.data[field]

            content_type = value.WhichOneof('content')

            # this if else statement need to be refactored it is too long
            # the check should be delegated to the type level
            content_type_dict = dict(
                tensor=Tensor,
                torch_tensor=TorchTensor,
                embedding=Embedding,
                any_url=AnyUrl,
                text_url=TextUrl,
                image_url=ImageUrl,
                id=ID,
            )
            if content_type in content_type_dict:
                fields[field] = content_type_dict[content_type].from_protobuf(
                    getattr(value, content_type)
                )
            elif content_type == 'text':
                fields[field] = value.text
            elif content_type == 'nested':
                fields[field] = cls._get_nested_document_class(field).from_protobuf(
                    value.nested
                )  # we get to the parent class
            elif content_type == 'chunks':
                from docarray import DocumentArray

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
