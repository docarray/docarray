import os
from typing import Type, Union
from uuid import UUID

from pydantic import BaseModel, Field

from docarray.document.abstract_document import AbstractDocument
from docarray.document.base_node import BaseNode

from .mixins import ProtoMixin


class BaseDocument(BaseModel, ProtoMixin, AbstractDocument, BaseNode):
    """
    The base class for Document
    """

    id: Union[int, str, UUID] = Field(default_factory=lambda: os.urandom(16).hex())

    @classmethod
    def _get_nested_document_class(cls, field: str) -> Type[BaseNode]:
        """
        Accessing the nested python Class define in the schema. Could be useful for
        reconstruction of Document in serialization/deserilization
        :param field: name of the field
        :return:
        """
        return cls.__fields__[field].type_
