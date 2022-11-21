from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseConfig, parse_obj_as
from pydantic.fields import ModelField

from docarray.proto import NodeProto

T = TypeVar("T")


class AbstractTyping(ABC):
    @abstractmethod
    def __get_validators__(cls):
        yield cls.validate

    @abstractmethod
    def validate(
        cls: Type[T],
        value: Union[str, int, UUID],
        field: Optional['ModelField'] = None,
        config: Optional['BaseConfig'] = None,
    ) -> T:
        return T

    @abstractmethod
    def _to_node_protobuf(self) -> NodeProto:
        return NodeProto(id=self)

    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)
