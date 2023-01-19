from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import BaseConfig
from pydantic.fields import ModelField

from docarray.base_document.base_node import BaseNode

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T')


class AbstractType(BaseNode):
    _proto_type_name: str

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    @abstractmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        ...

    @classmethod
    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        ...

    @abstractmethod
    def _to_node_protobuf(self: T) -> 'NodeProto':
        ...
