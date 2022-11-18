import os
from json import JSONEncoder
from typing import Type

from pydantic import BaseModel, Field

from docarray.document.abstract_document import AbstractDocument
from docarray.document.base_node import BaseNode
from docarray.typing import ID

from .mixins import ProtoMixin


class _DocumentJsonEncoder(JSONEncoder):
    """
    This is a custom JSONEncoder that will call the
    _to_json_compatible method of type. This Encoder will be
    used when calling doc.json()
    """

    def default(self, obj):
        if hasattr(obj, '_to_json_compatible'):
            return obj._to_json_compatible()
        return JSONEncoder.default(self, obj)


class BaseDocument(BaseModel, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: ID.validate(os.urandom(16).hex()))

    class Config:
        json_loads = _DocumentJsonEncoder

    @classmethod
    def _get_nested_document_class(cls, field: str) -> Type['BaseDocument']:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].type_
