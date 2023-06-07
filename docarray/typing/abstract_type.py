from abc import abstractmethod
from typing import Type, TypeVar, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from docarray.base_doc.base_node import BaseNode

T = TypeVar('T')


class AbstractType(BaseNode):

    ## TODO: this can stay compatible with pydantic v1
    @classmethod
    @abstractmethod
    def validate(cls: Type[T], __input_value: Any) -> T:
        ...

    @classmethod
    @abstractmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        ...