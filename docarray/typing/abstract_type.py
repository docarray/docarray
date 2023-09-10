from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Type, TypeVar

from docarray.utils._internal.pydantic import is_pydantic_v2

if TYPE_CHECKING:
    if is_pydantic_v2:
        from pydantic import GetCoreSchemaHandler
        from pydantic_core import core_schema

from docarray.base_doc.base_node import BaseNode

T = TypeVar('T')


class AbstractType(BaseNode):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    @abstractmethod
    def _docarray_validate(cls: Type[T], value: Any) -> T:
        ...

    if is_pydantic_v2:

        @classmethod
        def validate(cls: Type[T], value: Any, _: Any) -> T:
            return cls._docarray_validate(value)

    else:

        @classmethod
        def validate(
            cls: Type[T],
            value: Any,
        ) -> T:
            return cls._docarray_validate(value)

    if is_pydantic_v2:

        @classmethod
        @abstractmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: 'GetCoreSchemaHandler'
        ) -> 'core_schema.CoreSchema':
            ...
