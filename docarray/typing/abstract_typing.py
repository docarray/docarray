from abc import ABC, abstractmethod
from typing import Optional, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseConfig
from pydantic.fields import ModelField

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
        ...

    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        ...
