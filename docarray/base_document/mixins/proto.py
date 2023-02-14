from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Type, TypeVar

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode
from docarray.typing.proto_register import _PROTO_TYPE_NAME_TO_CLASS

if TYPE_CHECKING:
    from docarray.proto import DocumentProto, NodeProto


T = TypeVar('T', bound='ProtoMixin')


class ProtoMixin(AbstractDocument, BaseNode):
    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentProto') -> T:
        """create a Document from a protobuf message

        :param pb_msg: the proto message of the Document
        :return: a Document initialize with the proto data
        """

        return cls.from_protobuf_field_map(pb_msg)

    @classmethod
    def from_protobuf_field_map(
        cls: Type[T],
        pb_msg: 'DocumentProto',
        field_map: Optional[Mapping[str, str]] = None,
    ) -> T:
        """Create a Document from a protobuf message. You can optionally specify a mapping
        that will be used to map keys in the protobuf definition to keys (names) of the Document fields.

        .. code-block:: python
            from docarray import BaseDocument
            from docarray.typing import AnyUrl, TorchTensor

            import torch


            class A(BaseDocument):
                url: AnyUrl
                tensor: TorchTensor


            class B(BaseDocument):
                link: AnyUrl
                array: TorchTensor


            a = A(url='hello', tensor=torch.zeros(3))

            b = B.from_protobuf_with_schema_map(
                a.to_protobuf(), field_map={'url': 'link', 'tensor': 'array'}
            )

            assert b.link == a.url
            assert (b.array == a.tensor).all()

        :param pb_msg: The proto message representing the Document
        :param field_map: Optional mapping from proto fields to Document fields
        :return: A new Document initialized with the data from `pb_msg`
        """

        fields: Dict[str, Any] = {}

        field_map = field_map or {}

        for field_name in pb_msg.data:

            if field_name not in cls.__fields__.keys():
                if field_name in field_map.keys():
                    field_name_mapped = field_map[field_name]
                else:
                    continue  # optimization we don't even load the data if the key does not
                    # match any field in the cls or in the mapping
            else:
                field_name_mapped = field_name

            value = pb_msg.data[field_name]

            fields[field_name_mapped] = cls._get_content_from_node_proto(
                value, field_name_mapped
            )

        return cls(**fields)

    @classmethod
    def _get_content_from_node_proto(cls, value: 'NodeProto', field_name):
        """
        load the proto data from a node proto

        :param value: the proto node value
        :param field_name: the name of the field
        :return: the loaded field
        """
        content_type_dict = _PROTO_TYPE_NAME_TO_CLASS
        arg_to_container: Dict[str, Callable] = {
            'list': lambda x: x,
            'set': set,
            'tuple': tuple,
            'dict': dict,
        }

        content_key = value.WhichOneof('content')
        content_type = (
            value.type if value.WhichOneof('docarray_type') is not None else None
        )

        return_field: Any

        if content_type in content_type_dict:
            return_field = content_type_dict[content_type].from_protobuf(
                getattr(value, content_key)
            )
        elif content_key == 'document':
            return_field = cls._get_field_type(field_name).from_protobuf(
                value.document
            )  # we get to the parent class
        elif content_key == 'document_array':
            from docarray import DocumentArray

            return_field = DocumentArray.from_protobuf(
                value.document_array
            )  # we get to the parent class
        elif content_key is None:
            return_field = None
        elif content_type is None:

            if content_key in ['text', 'blob', 'integer', 'float', 'boolean']:
                return_field = getattr(value, content_key)

            elif content_key in arg_to_container.keys():
                from google.protobuf.json_format import MessageToDict

                return_field = arg_to_container[content_key](
                    MessageToDict(getattr(value, content_key))
                )

            else:
                raise ValueError(
                    f'key {content_key} is not supported for' f' deserialization'
                )

        else:
            raise ValueError(
                f'type {content_type}, with key {content_key} is not supported for'
                f' deserialization'
            )

        return return_field

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

                elif isinstance(value, str):
                    nested_item = NodeProto(text=value)

                elif isinstance(value, bool):
                    nested_item = NodeProto(boolean=value)

                elif isinstance(value, int):
                    nested_item = NodeProto(integer=value)

                elif isinstance(value, float):
                    nested_item = NodeProto(float=value)

                elif isinstance(value, bytes):
                    nested_item = NodeProto(blob=value)

                elif isinstance(value, list):
                    from google.protobuf.struct_pb2 import ListValue

                    lvalue = ListValue()
                    for item in value:
                        lvalue.append(item)
                    nested_item = NodeProto(list=lvalue)

                elif isinstance(value, set):
                    from google.protobuf.struct_pb2 import ListValue

                    lvalue = ListValue()
                    for item in value:
                        lvalue.append(item)
                    nested_item = NodeProto(set=lvalue)

                elif isinstance(value, tuple):
                    from google.protobuf.struct_pb2 import ListValue

                    lvalue = ListValue()
                    for item in value:
                        lvalue.append(item)
                    nested_item = NodeProto(tuple=lvalue)

                elif isinstance(value, dict):
                    from google.protobuf.struct_pb2 import Struct

                    struct = Struct()
                    struct.update(value)
                    nested_item = NodeProto(dict=struct)
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
