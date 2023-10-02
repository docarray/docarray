from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import parse_obj_as

from docarray.typing.abstract_type import AbstractType
from docarray.utils._internal.pydantic import bytes_validator, is_pydantic_v2

if is_pydantic_v2:
    from pydantic_core import core_schema

if TYPE_CHECKING:
    from docarray.proto import NodeProto

    if is_pydantic_v2:
        from pydantic import GetCoreSchemaHandler

T = TypeVar('T', bound='BaseBytes')


class BaseBytes(bytes, AbstractType):
    """
    Bytes type for docarray
    """

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Any,
    ) -> T:
        value = bytes_validator(value)
        return cls(value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    if is_pydantic_v2:

        @classmethod
        @abstractmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: 'GetCoreSchemaHandler'
        ) -> 'core_schema.CoreSchema':
            return core_schema.general_after_validator_function(
                cls.validate,
                core_schema.bytes_schema(),
            )
