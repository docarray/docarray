import os
from typing import TYPE_CHECKING, Any, Type, TypeVar

import orjson
from pydantic import BaseModel, Field, parse_obj_as
from rich.console import Console

from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import IOMixin, UpdateMixin
from docarray.typing import ID

if TYPE_CHECKING:
    from docarray.array.stacked.columnstorage import ColumnStorageView

_console: Console = Console()

T = TypeVar('T', bound='BaseDocument')


class BaseDocument(BaseModel, IOMixin, UpdateMixin, BaseNode):
    """
    The base class for Documents
    """

    id: ID = Field(default_factory=lambda: parse_obj_as(ID, os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def from_view(cls: Type[T], storage_view: 'ColumnStorageView') -> T:
        doc = cls.__new__(cls)
        object.__setattr__(doc, '__dict__', storage_view)
        doc._init_private_attributes()
        return doc

    @classmethod
    def _get_field_type(cls, field: str) -> Type:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].outer_type_

    def __str__(self) -> str:
        with _console.capture() as capture:
            _console.print(self)

        return capture.get().strip()

    def summary(self) -> None:
        """Print non-empty fields and nested structure of this Document object."""
        from docarray.display.document_summary import DocumentSummary

        DocumentSummary(doc=self).summary()

    @classmethod
    def schema_summary(cls) -> None:
        """Print a summary of the Documents schema."""
        from docarray.display.document_summary import DocumentSummary

        DocumentSummary.schema_summary(cls)

    def _ipython_display_(self) -> None:
        """Displays the object in IPython as a summary"""
        self.summary()

    def is_view(self) -> bool:
        from docarray.array.stacked.columnstorage import ColumnStorageView

        return isinstance(self.__dict__, ColumnStorageView)

    def __getattr__(self, item) -> Any:
        return self.__dict__[item]
