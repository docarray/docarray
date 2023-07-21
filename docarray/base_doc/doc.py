import os
import warnings
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    no_type_check,
)

import orjson
from pydantic import BaseModel, Field
from pydantic.main import ROOT_KEY
from rich.console import Console

from docarray.base_doc.base_node import BaseNode
from docarray.base_doc.io.json import orjson_dumps_and_decode
from docarray.base_doc.mixins import IOMixin, UpdateMixin
from docarray.typing import ID
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass

if TYPE_CHECKING:
    from pydantic import Protocol
    from pydantic.types import StrBytes
    from pydantic.typing import AbstractSetIntStr, DictStrAny, MappingIntStrAny

    from docarray.array.doc_vec.column_storage import ColumnStorageView

_console: Console = Console()

T = TypeVar('T', bound='BaseDoc')
T_update = TypeVar('T_update', bound='UpdateMixin')


ExcludeType = Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]


class BaseDoc(BaseModel, IOMixin, UpdateMixin, BaseNode):
    """
    BaseDoc is the base class for all Documents. This class should be subclassed
    to create new Document types with a specific schema.

    The schema of a Document is defined by the fields of the class.

    Example:
    ```python
    from docarray import BaseDoc
    from docarray.typing import NdArray, ImageUrl
    import numpy as np


    class MyDoc(BaseDoc):
        embedding: NdArray[512]
        image: ImageUrl


    doc = MyDoc(embedding=np.zeros(512), image='https://example.com/image.jpg')
    ```


    BaseDoc is a subclass of [pydantic.BaseModel](
    https://docs.pydantic.dev/usage/models/) and can be used in a similar way.
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
        _load_extra_fields_from_protobuf = False

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
        content: Any = None
        if self.is_view():
            attr_str = ", ".join(
                f"{field}={self.__getattr__(field)}" for field in self.__dict__.keys()
            )
            content = f"{self.__class__.__name__}({attr_str})"
        else:
            content = self

        with _console.capture() as capture:
            _console.print(content)

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
        if not isinstance(other, BaseDoc):
            return False

        if self.__fields__.keys() != other.__fields__.keys():
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
    ### this section is just for documentation purposes will be removed later once
    # https://github.com/mkdocstrings/griffe/issues/138 is fixed ##############
    ########################################################################################################################################################

    def json(
        self,
        *,
        include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']] = None,
        exclude: ExcludeType = None,
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
        Generate a JSON representation of the model, `include` and `exclude`
        arguments as per `dict()`.

        `encoder` is an optional function to supply as `default` to json.dumps(),
        other arguments as per `json.dumps()`.
        """
        exclude, original_exclude, doclist_exclude_fields = self._exclude_docarray(
            exclude=exclude
        )

        # this is copy from pydantic code
        if skip_defaults is not None:
            warnings.warn(
                f'{self.__class__.__name__}.json(): "skip_defaults" is deprecated and replaced by "exclude_unset"',
                DeprecationWarning,
            )
            exclude_unset = skip_defaults
        encoder = cast(Callable[[Any], Any], encoder or self.__json_encoder__)

        # We don't directly call `self.dict()`, which does exactly this with `to_dict=True`
        # because we want to be able to keep raw `BaseModel` instances and not as `dict`.
        # This allows users to write custom JSON encoders for given `BaseModel` classes.
        data = dict(
            self._iter(
                to_dict=models_as_dict,
                by_alias=by_alias,
                include=include,
                exclude=exclude,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            )
        )

        # this is the custom part to deal with DocList
        for field in doclist_exclude_fields:
            # we need to do this because pydantic will not recognize DocList correctly
            original_exclude = original_exclude or {}
            if field not in original_exclude:
                data[field] = getattr(
                    self, field
                )  # here we need to keep doclist as doclist otherwise if a user want to have a special json config it will not work

        # this is copy from pydantic code
        if self.__custom_root_type__:
            data = data[ROOT_KEY]
        return self.__config__.json_dumps(data, default=encoder, **dumps_kwargs)

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

    def dict(
        self,
        *,
        include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']] = None,
        exclude: ExcludeType = None,
        by_alias: bool = False,
        skip_defaults: Optional[bool] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> 'DictStrAny':
        """
        Generate a dictionary representation of the model, optionally specifying
        which fields to include or exclude. The method also includes the attributes
        and their values when attributes are objects of class types which can include
        nesting. This method differs from the `dict()` method of python which only
        prints the methods and attributes of the object and only the type of the
        attribute when attributes are types of other class.

        """

        exclude, original_exclude, docarray_exclude_fields = self._exclude_docarray(
            exclude=exclude
        )

        data = super().dict(
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

        for field in docarray_exclude_fields:
            # we need to do this because pydantic will not recognize DocList correctly
            original_exclude = original_exclude or {}
            if field not in original_exclude:
                val = getattr(self, field)
                data[field] = [doc.dict() for doc in val] if val is not None else None

        return data

    def _exclude_docarray(
        self, exclude: ExcludeType
    ) -> Tuple[ExcludeType, ExcludeType, List[str]]:
        docarray_exclude_fields = []
        for field in self.__fields__.keys():
            from docarray import DocList, DocVec

            type_ = self._get_field_type(field)
            if isinstance(type_, type) and (
                safe_issubclass(type_, DocList) or safe_issubclass(type_, DocVec)
            ):
                docarray_exclude_fields.append(field)

        original_exclude = exclude
        if exclude is None:
            exclude = set(docarray_exclude_fields)
        elif isinstance(exclude, AbstractSet):
            exclude = set([*exclude, *docarray_exclude_fields])
        elif isinstance(exclude, Mapping):
            exclude = dict(**exclude)
            exclude.update({field: ... for field in docarray_exclude_fields})

        return (
            exclude,
            original_exclude,
            docarray_exclude_fields,
        )

    to_json = json
