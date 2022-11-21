import os
from typing import Type

import orjson
from pydantic import BaseModel, Field

from docarray.document.abstract_document import AbstractDocument
from docarray.document.base_node import BaseNode
from docarray.typing import ID

from .mixins import ProtoMixin


def _default_orjson(obj):
    """
    default option for orjson dumps. It will call _to_json_compatible
    from docarray typing object that expose such method.
    :param obj:
    :return: return a json compatible object
    """

    if getattr(obj, '_to_json_compatible'):
        return obj._to_json_compatible()


def _orjson_dumps(v, *, default):
    # orjson.dumps returns bytes, to match standard json.dumps we need to decode
    return orjson.dumps(
        v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY
    ).decode()


class BaseDocument(BaseModel, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: ID.validate(os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = _orjson_dumps

    @classmethod
    def _get_nested_document_class(cls, field: str) -> Type['BaseDocument']:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].type_
