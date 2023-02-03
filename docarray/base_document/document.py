import os
from typing import Type

import orjson
from pydantic import BaseModel, Field, parse_obj_as
from rich.console import Console

from docarray.base_document.abstract_document import AbstractDocument
from docarray.base_document.base_node import BaseNode
from docarray.base_document.io.json import orjson_dumps, orjson_dumps_and_decode
from docarray.base_document.mixins import PlotMixin, ProtoMixin
from docarray.typing import ID

_console: Console = Console()


class BaseDocument(BaseModel, PlotMixin, ProtoMixin, AbstractDocument, BaseNode):
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

    def _get_string_for_regex_filter(self):
        return str(self)

    def update(self, other: 'BaseDocument'):
        """
        Updates the content of this Document with the contents of other using
        :func:`~docarray.utils.reduce.reduce_docs`.

        It behaves as an update operation for Dictionaries, except that since
        it is applied to a static schema type, the presence of the field is
        given by the field not having a None value.

            EXAMPLE USAGE

            .. code-block:: python

                from docarray import BaseDocument
                from docarray.documents import Text

                class MyDocument(BaseDocument):
                    content: str
                    title: Optional[str] = None
                    tags_: List

                doc1 = MyDocument(content='Core content of the document',
                    title='Title', tags_=['python', 'AI'])
                doc2 = MyDocument(content='Core content updated', tags_=['docarray'])

                doc1.update(doc2)
                assert doc1.content == 'Core content updated'
                assert doc1.title == 'Title'
                assert doc1.tags_ == ['python', 'AI', 'docarray']

        :param other: The Document with which to update the contents of this
        """
        from docarray.utils.reduce import reduce_docs

        reduce_docs(self, other)
