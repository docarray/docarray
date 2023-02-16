import os
from typing import TYPE_CHECKING, Optional, Type

import orjson
from pydantic import BaseModel, Field, PrivateAttr, parse_obj_as
from rich.console import Console

from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import IOMixin, UpdateMixin
from docarray.typing import ID

if TYPE_CHECKING:
    from docarray.array.array_stacked import DocumentArrayStacked

_console: Console = Console()


class BaseDocument(BaseModel, IOMixin, UpdateMixin, BaseNode):
    """
    The base class for Documents
    TODO add some documentation here this is the most important place
    """

    id: ID = Field(default_factory=lambda: parse_obj_as(ID, os.urandom(16).hex()))
    _da_ref: Optional['DocumentArrayStacked'] = PrivateAttr(None)

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def _get_field_type(cls, field: str) -> Type:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].outer_type_

    def __str__(self):
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

    def _ipython_display_(self):
        """Displays the object in IPython as a summary"""
        self.summary()

    def _is_inside_da_stack(self) -> bool:
        """
        :return: return true if the Document is inside a DocumentArrayStack
        """
        return self._da_ref is not None
