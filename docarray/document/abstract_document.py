from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Iterable, Type

from pydantic.fields import ModelField

if TYPE_CHECKING:
    from docarray.document import BaseNode


class AbstractDocument(Iterable):
    __fields__: Dict[str, ModelField]

    @classmethod
    @abstractmethod
    def _get_nested_document_class(cls, field: str) -> Type['BaseNode']:
        ...
