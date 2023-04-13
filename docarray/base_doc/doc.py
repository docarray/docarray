import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
    no_type_check,
)

import orjson
from pydantic import BaseModel, Field
from rich.console import Console

from docarray.base_doc.base_node import BaseNode
from docarray.base_doc.io.json import orjson_dumps_and_decode
from docarray.base_doc.mixins import IOMixin, UpdateMixin
from docarray.typing import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from pydantic import Protocol
    from pydantic.types import StrBytes
    from pydantic.typing import AbstractSetIntStr, MappingIntStrAny

    from docarray.array.doc_vec.column_storage import ColumnStorageView

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

    def json(
        self,
        *,
        include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']] = None,
        exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']] = None,
        by_alias: bool = False,
        skip_defaults: Optional[bool] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        encoder: Optional[Callable[[Any], Any]] = None,
        models_as_dict: bool = True,
        **dumps_kwargs: Any,
    ) -> str:
        """
        Generate a JSON representation of the model, `include` and `exclude` arguments as per `dict()`.

        `encoder` is an optional function to supply as `default` to json.dumps(), other arguments as per `json.dumps()`.
        """
        return super().json(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            encoder=encoder,
            models_as_dict=models_as_dict,
            **dumps_kwargs,
        )

    @no_type_check
    @classmethod
    def parse_raw(
        cls: Type[T],
        b: 'StrBytes',
        *,
        content_type: str = None,
        encoding: str = 'utf8',
        proto: 'Protocol' = None,
        allow_pickle: bool = False,
    ) -> T:
        """
        Parse a raw string or bytes into a base doc
        :param b:
        :param content_type:
        :param encoding: the encoding to use when parsing a string, defaults to 'utf8'
        :param proto: protocol to use.
        :param allow_pickle: allow pickle protocol
        :return: a document
        """
        return super(BaseDoc, cls).parse_raw(
            b,
            content_type=content_type,
            encoding=encoding,
            proto=proto,
            allow_pickle=allow_pickle,
        )
