import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar

import orjson
from pydantic import BaseModel, Field
from rich.console import Console

from docarray.base_doc.base_node import BaseNode
from docarray.base_doc.io.json import orjson_dumps_and_decode
from docarray.base_doc.mixins import IOMixin, UpdateMixin
from docarray.typing import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.array.doc_vec.column_storage import ColumnStorageView
    from docarray.proto import DocProto

_console: Console = Console()

T = TypeVar('T', bound='BaseDoc')
T_update = TypeVar('T_update', bound='UpdateMixin')


class BaseDoc(BaseModel, IOMixin, UpdateMixin, BaseNode):
    """
    The base class for Documents
    """

    id: Optional[ID] = Field(default_factory=lambda: ID(os.urandom(16).hex()))

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps_and_decode
        # `DocArrayResponse` is able to handle tensors by itself.
        # Therefore, we stop FastAPI from doing any transformations
        # on tensors by setting an identity function as a custom encoder.
        json_encoders = {AbstractTensor: lambda x: x}

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
        from docarray.array.doc_vec.column_storage import ColumnStorageView

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
        if self.dict().keys() != other.dict().keys():
            return False

        for field_name in self.__fields__:
            value1 = getattr(self, field_name)
            value2 = getattr(other, field_name)

            if field_name == 'id':
                continue

            if isinstance(value1, AbstractTensor) and isinstance(
                value2, AbstractTensor
            ):
                comp_be1 = value1.get_comp_backend()
                comp_be2 = value2.get_comp_backend()

                if comp_be1.shape(value1) != comp_be2.shape(value2):
                    return False
                if (
                    not (comp_be1.to_numpy(value1) == comp_be2.to_numpy(value2))
                    .all()
                    .item()
                ):
                    return False
            else:
                if value1 != value2:
                    return False
        return True

    def __ne__(self, other) -> bool:
        return not (self == other)

    def _docarray_to_json_compatible(self) -> Dict:
        """
        Convert itself into a json compatible object
        :return: A dictionary of the BaseDoc object
        """
        return self.dict()

    ########################################################################################################################################################
    ### this section is just for documentation purposes will be removed later once https://github.com/mkdocstrings/griffe/issues/138 is fixed ##############
    ########################################################################################################################################################

    def to_bytes(
        self, protocol: str = 'protobuf', compress: Optional[str] = None
    ) -> bytes:
        """Serialize itself into bytes.

        For more Pythonic code, please use ``bytes(...)``.

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress algorithm to use
        :return: the binary serialization in bytes
        """
        return super().to_bytes(protocol, compress)

    @classmethod
    def from_bytes(
        cls: Type[T],
        data: bytes,
        protocol: str = 'protobuf',
        compress: Optional[str] = None,
    ) -> T:
        """Build Document object from binary bytes

        :param data: binary bytes
        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress method to use
        :return: a Document object
        """
        return super(BaseDoc, cls).from_bytes(data, protocol, compress)

    def to_base64(
        self, protocol: str = 'protobuf', compress: Optional[str] = None
    ) -> str:
        """Serialize a Document object into as base64 string

        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress method to use
        :return: a base64 encoded string
        """
        return super().to_base64(protocol, compress)

    @classmethod
    def from_base64(
        cls: Type[T],
        data: str,
        protocol: str = 'pickle',
        compress: Optional[str] = None,
    ) -> T:
        """Build Document object from binary bytes

        :param data: a base64 encoded string
        :param protocol: protocol to use. It can be 'pickle' or 'protobuf'
        :param compress: compress method to use
        :return: a Document object
        """
        return super(BaseDoc, cls).from_base64(data, protocol, compress)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: 'DocProto') -> T:
        """create a Document from a protobuf message

        :param pb_msg: the proto message of the Document
        :return: a Document initialize with the proto data
        """
        return super(BaseDoc, cls).from_protobuf(pb_msg)

    def update(self, other: T_update):
        """
        Updates self with the content of other. Changes are applied to self.
        Updating one Document with another consists in the following:
         - setting data properties of the second Document to the first Document
         if they are not None
         - Concatenating lists and updating sets
         - Updating recursively Documents and DocArrays
         - Updating Dictionaries of the left with the right

        It behaves as an update operation for Dictionaries, except that since
        it is applied to a static schema type, the presence of the field is
        given by the field not having a None value and that DocArrays,
        lists and sets are concatenated. It is worth mentioning that Tuples
        are not merged together since they are meant to be inmutable,
        so they behave as regular types and the value of `self` is updated
        with the value of `other`


        ---

        ```python
        from docarray import BaseDoc
        from docarray.documents import Text


        class MyDocument(BaseDoc):
            content: str
            title: Optional[str] = None
            tags_: List


        doc1 = MyDocument(
            content='Core content of the document', title='Title', tags_=['python', 'AI']
        )
        doc2 = MyDocument(content='Core content updated', tags_=['docarray'])

        doc1.update(doc2)
        assert doc1.content == 'Core content updated'
        assert doc1.title == 'Title'
        assert doc1.tags_ == ['python', 'AI', 'docarray']
        ```

        ---
        :param other: The Document with which to update the contents of this
        """
        super().update(other)
