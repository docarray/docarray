import os
from typing import Type

import orjson
from pydantic import BaseModel, Field

from docarray.document.abstract_document import AbstractDocument
from docarray.document.base_node import BaseNode
from docarray.document.io.json import orjson_dumps
from docarray.document.mixins import ProtoMixin
from docarray.typing import ID


class BaseDocument(BaseModel, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: ID.validate(os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    @classmethod
    def _get_nested_document_class(cls, field: str) -> Type['BaseDocument']:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].type_
