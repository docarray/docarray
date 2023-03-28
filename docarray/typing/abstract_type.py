from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar

from pydantic import BaseConfig
from pydantic.fields import ModelField


T = TypeVar('T')


class AbstractType(ABC):
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
