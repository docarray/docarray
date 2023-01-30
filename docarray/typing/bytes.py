from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import parse_obj_as
from pydantic.validators import bytes_validator

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='Bytes')


@_register_proto(proto_type_name='bytes')
class Bytes(AbstractType, bytes):
    """
    Represent a byte object
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        if isinstance(value, cls):
            return value
        else:
            value = bytes_validator(value)
            return cls(value)

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert an ID into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
        """
        read base64 from a proto msg
        :param pb_msg:
        :return: a string
        """
        return parse_obj_as(cls, pb_msg)

    def _docarray_to_json_compatible(self) -> str:
        """
        Convert base64 into a json compatible object
        :return: a representation of the tensor compatible with orjson
        """
        return self.decode()
