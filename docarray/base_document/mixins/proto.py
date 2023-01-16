from typing import TYPE_CHECKING, Any, Dict, Type, TypeVar

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode

if TYPE_CHECKING:
    from docarray.proto import DocumentProto, NodeProto


try:
    import torch  # noqa: F401
except ImportError:
    torch_imported = False
else:
    from docarray.typing.tensor.torch_tensor import TorchTensor

    torch_imported = True


T = TypeVar('T', bound='ProtoMixin')


class ProtoMixin(AbstractDocument, BaseNode):
    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentProto') -> T:
        """create a Document from a protobuf message"""
        from docarray.typing import (  # TorchTensor,
            ID,
            AnyEmbedding,
            AnyUrl,
            ImageUrl,
            Mesh3DUrl,
            NdArray,
            PointCloud3DUrl,
            TextUrl,
        )

        fields: Dict[str, Any] = {}

        for field in pb_msg.data:
            value = pb_msg.data[field]

            content_type = value.WhichOneof('content')

            # this if else statement need to be refactored it is too long
            # the check should be delegated to the type level
            content_type_dict = dict(
                ndarray=NdArray,
                embedding=AnyEmbedding,
                any_url=AnyUrl,
                text_url=TextUrl,
                image_url=ImageUrl,
                mesh_url=Mesh3DUrl,
                point_cloud_url=PointCloud3DUrl,
                id=ID,
            )

            if torch_imported:
                content_type_dict['torch_tensor'] = TorchTensor

            if content_type in content_type_dict:
                fields[field] = content_type_dict[content_type].from_protobuf(
                    getattr(value, content_type)
                )
            elif content_type == 'text':
                fields[field] = value.text
            elif content_type == 'nested':
                fields[field] = cls._get_field_type(field).from_protobuf(
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

        return cls.construct(**fields)

    def to_protobuf(self) -> 'DocumentProto':
        """Convert Document into a Protobuf message.

        :return: the protobuf message
        """
        from docarray.proto import DocumentProto, NodeProto

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

    def _to_node_protobuf(self) -> 'NodeProto':
        from docarray.proto import NodeProto

        """Convert Document into a NodeProto protobuf message. This function should be
        called when the Document is nest into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(document=self.to_protobuf())
