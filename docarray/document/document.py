import os
from typing import Union
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

    class Config:
        arbitrary_types_allowed = True
