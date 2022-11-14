from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union
from uuid import UUID

from docarray.document.base_node import BaseNode
from docarray.proto import NodeProto

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


T = TypeVar('T', bound='ID')


class ID(str, BaseNode):
    """
    Represent an unique ID
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, int, UUID],
        field: Optional['ModelField'] = None,
        config: Optional['BaseConfig'] = None,
    ) -> T:

        try:
            id: str = str(value)
            return cls(id)
        except Exception:
            raise ValueError(f'Expected a str, int or UUID, got {type(value)}')

    def _to_node_protobuf(self) -> NodeProto:
        """Convert an ID into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        return NodeProto(id=self)
