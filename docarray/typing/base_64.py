import base64
from typing import TYPE_CHECKING, Any, Type, TypeVar

from docarray.typing.bytes import Bytes
from docarray.typing.proto_register import _register_proto

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='Base64')


@_register_proto(proto_type_name='base64')
class Base64(Bytes):
    """
    Represent a base64 objects
    """

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
            value = super().validate(value, field, config)
            return cls(base64.b64encode(value))

    def decode(self) -> bytes:
        """
        Decode the base64 byte into corresponding bytes
        :return:
        """
        return base64.b64decode(self)

    def decode_str(self) -> str:
        """
        Decode the base64 byte into corresponding string
        :return:
        """
        return self.decode().decode()
