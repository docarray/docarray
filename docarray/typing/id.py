from typing import TYPE_CHECKING, Any, Type, TypeVar, Union
from uuid import UUID

from pydantic import parse_obj_as

from docarray.typing.proto_register import _register_proto
from docarray.utils._internal.pydantic import is_pydantic_v2

if TYPE_CHECKING:
    from docarray.proto import NodeProto

from docarray.typing.abstract_type import AbstractType

if is_pydantic_v2:
    from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import core_schema

T = TypeVar('T', bound='ID')


@_register_proto(proto_type_name='id')
class ID(str, AbstractType):
    """
    Represent an unique ID
    """

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Union[str, int, UUID],
    ) -> T:
        try:
            id: str = str(value)
            return cls(id)
        except Exception:
            raise ValueError(f'Expected a str, int or UUID, got {type(value)}')

    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert an ID into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(text=self, type=self._proto_type_name)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'str') -> T:
        """
        read ndarray from a proto msg
        :param pb_msg:
        :return: a string
        """
        return parse_obj_as(cls, pb_msg)

    if is_pydantic_v2:

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: Type[Any], handler: 'GetCoreSchemaHandler'
        ) -> core_schema.CoreSchema:
            return core_schema.general_plain_validator_function(
                cls.validate,
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            field_schema: dict[str, Any] = {}
            field_schema.update(type='string')
            return field_schema
