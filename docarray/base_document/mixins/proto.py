from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Type, TypeVar

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode
from docarray.typing.proto_register import _PROTO_TYPE_NAME_TO_CLASS

if TYPE_CHECKING:
    from docarray.proto import DocumentProto, NodeProto


T = TypeVar('T', bound='ProtoMixin')


class ProtoMixin(AbstractDocument, BaseNode):
    @classmethod
    def _get_data_dict_from_proto(
        cls: Type[T], pb_msg: 'DocumentProto'
    ) -> Dict[str, Any]:
        """
        read the proto and load the data into a dict. Need to be use before calling
        the validation of pydantic.
        :return: a dict with the data before validation
        """
        fields: Dict[str, Any] = {}

        for field in pb_msg.data:
            value = pb_msg.data[field]

            content_type_dict = _PROTO_TYPE_NAME_TO_CLASS

            content_key = value.WhichOneof('content')
            content_type = (
                value.type if value.WhichOneof('docarray_type') is not None else None
            )

            if content_type in content_type_dict:
                fields[field] = content_type_dict[content_type].from_protobuf(
                    getattr(value, content_key)
                )
            elif content_key == 'document':
                fields[field] = cls._get_field_type(field).from_protobuf(
                    value.document
                )  # we get to the parent class
            elif content_key == 'document_array':
                from docarray import DocumentArray

                fields[field] = DocumentArray.from_protobuf(
                    value.document_array
                )  # we get to the parent class
            elif content_key is None:
                fields[field] = None
            elif content_type is None:
                if content_key == 'text':
                    fields[field] = value.text
                elif content_key == 'blob':
                    fields[field] = value.blob
                elif content_key == 'integer':
                    fields[field] = value.integer
                elif content_key == 'float':
                    fields[field] = value.float
                elif content_key == 'boolean':
                    fields[field] = value.boolean
                elif content_key == 'list':
                    from google.protobuf.json_format import MessageToDict

                    fields[field] = MessageToDict(value.list)
                elif content_key == 'set':
                    from google.protobuf.json_format import MessageToDict

                    fields[field] = set(MessageToDict(value.set))
                elif content_key == 'tuple':
                    from google.protobuf.json_format import MessageToDict

                    fields[field] = tuple(MessageToDict(value.tuple))
                elif content_key == 'dict':
                    from google.protobuf.json_format import MessageToDict

                    fields[field] = MessageToDict(value.dict)
                else:
                    raise ValueError(
                        f'key {content_key} is not supported for' f' deserialization'
                    )

            else:
                raise ValueError(
                    f'type {content_type}, with key {content_key} is not supported for'
                    f' deserialization'
                )

        return fields

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocumentProto') -> T:
        """create a dict with field and value from proto"""
        return cls(**cls._get_data_dict_from_proto(pb_msg))

    @classmethod
    def from_protobuf_smart(
        cls: Type[T],
        pb_msg: 'DocumentProto',
        cast_map: Optional[Mapping[str, str]] = None,
    ) -> T:
        """create a Document from a protobuf message"""
        # TODO this could be optimized. Here we are first loading the data then from the
        # proto and then only we remove the useless one. We could optimize here
        # and do the same mechanism than for in the document but instead of checking
        # the data type we could check the `document_type` from the proto
        # and we would load only the necessary data
        return cls.smart_parse_obj(cls._get_data_dict_from_proto(pb_msg), cast_map)

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
