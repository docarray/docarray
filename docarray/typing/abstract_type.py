from abc import abstractmethod
from typing import Type, TypeVar

from pydantic_core import core_schema

from docarray.base_doc.base_node import BaseNode

T = TypeVar('T')


class AbstractType(BaseNode):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate


    ## TODO: this can stay compatible with pydantic v1
    @classmethod
    @abstractmethod
    def validate(cls: Type[T], __input_value: str, _: core_schema.ValidationInfo) -> T:
        ...