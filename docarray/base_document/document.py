import os
from typing import List, Type, Optional, TypeVar

import orjson
from pydantic import BaseModel, Field, parse_obj_as
from rich.console import Console
from typing_inspect import get_origin
import pickle
import base64

from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.utils.compress import _compress_bytes, _decompress_bytes
from docarray.base_document.mixins import ProtoMixin, UpdateMixin
from docarray.typing import ID

_console: Console = Console()

T = TypeVar('T', bound='ProtoMixin')


class BaseDocument(BaseModel, ProtoMixin, UpdateMixin, BaseNode):
    """
    The base class for Document
    """

    id: ID = Field(default_factory=lambda: parse_obj_as(ID, os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        json_encoders = {dict: orjson_dumps}

        validate_assignment = True

    @classmethod
    def _get_field_type(cls, field: str) -> Type['BaseDocument']:
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

    def to_bytes(self, protocol: str = 'pickle-array', compress: Optional[str] = None):
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param protocol: protocol to use
        :param compress: compress algorithm to use
        :return: the binary serialization in bytes
        """
        import pickle

        if protocol == 'pickle':
            bstr = pickle.dumps(self)
        elif protocol == 'protobuf':
            bstr = self.to_protobuf().SerializePartialToString()
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )
        return _compress_bytes(bstr, algorithm=compress)

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: str = 'pickle',
        compress: Optional[str] = None,
    ) -> T:
        """Build Document object from binary bytes

        :param data: binary bytes
        :param protocol: protocol to use
        :param compress: compress method to use
        :return: a Document object
        """
        bstr = _decompress_bytes(data, algorithm=compress)
        if protocol == 'pickle':
            return pickle.loads(bstr)
        elif protocol == 'protobuf':
            from docarray.proto import DocumentProto

            pb_msg = DocumentProto()
            pb_msg.ParseFromString(bstr)
            return cls.from_protobuf(pb_msg)
        else:
            raise ValueError(
                f'protocol={protocol} is not supported. Can be only `protobuf` or pickle protocols 0-5.'
            )

    def to_base64(
        self, protocol: str = 'pickle', compress: Optional[str] = None
    ) -> str:
        """Serialize a Document object into as base64 string

        :param protocol: protocol to use
        :param compress: compress method to use
        :return: a base64 encoded string
        """
        return base64.b64encode(self.to_bytes(protocol, compress)).decode('utf-8')

    @classmethod
    def from_base64(
        cls: Type[T],
        data: str,
        protocol: str = 'pickle',
        compress: Optional[str] = None,
    ) -> T:
        """Build Document object from binary bytes

        :param data: a base64 encoded string
        :param protocol: protocol to use
        :param compress: compress method to use
        :return: a Document object
        """
        return cls.from_bytes(base64.b64decode(data), protocol, compress)

  def _ipython_display_(self):
        """Displays the object in IPython as a summary"""
        self.summary()
