import os
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import orjson
from pydantic import BaseModel, Field
from rich.console import Console

import numpy as np

from docarray.base_document.base_node import BaseNode
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import IOMixin, UpdateMixin
from docarray.typing import ID

if TYPE_CHECKING:
    from docarray.array.stacked.column_storage import ColumnStorageView

_console: Console = Console()

T = TypeVar('T', bound='BaseDocument')


class BaseDocument(BaseModel, IOMixin, UpdateMixin, BaseNode):
    """
    The base class for Documents
    """

    id: Optional[ID] = Field(default_factory=lambda: ID(os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def from_view(cls: Type[T], storage_view: 'ColumnStorageView') -> T:
        doc = cls.__new__(cls)
        object.__setattr__(doc, '__dict__', storage_view)
        object.__setattr__(doc, '__fields_set__', set(storage_view.keys()))

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
        from docarray.array.stacked.column_storage import ColumnStorageView

        return isinstance(self.__dict__, ColumnStorageView)

    def __getattr__(self, item) -> Any:
        if item in self.__fields__.keys():
            return self.__dict__[item]
        else:
            return super().__getattribute__(item)

    def __setattr__(self, field, value) -> None:
        if not self.is_view():
            super().__setattr__(field, value)
        else:
            # here we first validate with pydantic
            # Then we apply the value to the remote dict,
            # and we change back the __dict__ value to the remote dict
            dict_ref = self.__dict__
            super().__setattr__(field, value)
            for key, val in self.__dict__.items():
                dict_ref[key] = val
            object.__setattr__(self, '__dict__', dict_ref)

    def __eq__(self, other) -> bool:
        if self.dict().keys() != other.dict().keys() :
            return False
        
        for key1,key2 in zip(self.dict(), other.dict()):
            
            value1 = self.dict()[key1]
            value2 = other.dict()[key2]
            
            if key1 == "id" and key2 == "id":
                continue

            if isinstance(value1, AbstractTensor) and isinstance(value2, AbstractTensor):
                comp_be = value1.get_comp_backend()
                if not comp_be.equal(value1, value2):
                    return False
            else:
                if value1 != value2 :
                    return False
        return True
    
    def __ne__(self,other) -> bool:
        return not (self == other)