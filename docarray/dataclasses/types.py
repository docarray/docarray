import base64
import typing
from dataclasses import (
    dataclass as _dataclass,
    Field as _Field,
    is_dataclass as _is_dataclass,
    field,
)
from enum import Enum
from pathlib import Path
from typing import TypeVar, ForwardRef, Callable, Optional, TYPE_CHECKING

from .deserializers import (
    image_deserializer,
    text_deserializer,
    audio_deserializer,
    json_deserializer,
)
from .serializers import (
    image_serializer,
    text_serializer,
    audio_serializer,
    json_serializer,
)

if TYPE_CHECKING:
    import scipy.sparse
    import tensorflow
    import torch
    import numpy as np
    from ..typing import T
    from docarray import Document
    from PIL.Image import Image as PILImage


class AttributeType(str, Enum):
    DOCUMENT = 'document'
    PRIMITIVE = 'primitive'
    ITERABLE_PRIMITIVE = 'iterable_primitive'
    ITERABLE_DOCUMENT = 'iterable_document'
    NESTED = 'nested'
    ITERABLE_NESTED = 'iterable_nested'


class Field(_Field):
    def __init__(
        self,
        serializer: Callable,
        deserializer: Callable,
        source_field: Optional[_Field] = None,
    ):
        if not source_field:
            source_field = field()

        self.name = source_field.name
        self.type = source_field.type
        self.default = source_field.default
        self.default_factory = source_field.default_factory
        self.init = source_field.init
        self.repr = source_field.repr
        self.hash = source_field.hash
        self.compare = source_field.compare
        self.metadata = source_field.metadata
        self._field_type = source_field._field_type

        self.serializer = serializer
        self.deserializer = deserializer


Image = TypeVar(
    'Image',
    ForwardRef('np.ndarray'),
    ForwardRef('tensorflow.Tensor'),
    ForwardRef('torch.Tensor'),
    str,
    ForwardRef('PILImage'),
)

Text = TypeVar('Text', bound=str)

Audio = TypeVar(
    'Audio',
    str,
    Path,
)

JSON = TypeVar('JSON', str, dict)

_TYPES_REGISTRY = {
    Image: lambda x: Field(image_serializer, image_deserializer, x),
    Text: lambda x: Field(text_serializer, text_deserializer, x),
    Audio: lambda x: Field(audio_serializer, audio_deserializer, x),
    JSON: lambda x: Field(json_serializer, json_deserializer, x),
}


def dataclass(cls: 'T' = None) -> 'T':
    """Extends python standard dataclass decorator to add functionalities to enable multi modality support to Document.
    Returns the same class as was passed in, with dunder methods and from_document method.


    If init is true, an __init__() method is added to the class. If
    repr is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method function is added. If frozen is true, fields may
    not be assigned to after instance creation.
    """

    def wrap(cls):
        decorated_cls = _dataclass(
            cls,
            init=True,
            repr=True,
            eq=True,
            order=False,
            unsafe_hash=False,
            frozen=False,
        )
        setattr(decorated_cls, from_document.__func__.__name__, from_document)
        for key, f in decorated_cls.__dataclass_fields__.items():
            if isinstance(f, Field):
                continue

            if f.type in _TYPES_REGISTRY:
                decorated_cls.__dataclass_fields__[key] = _TYPES_REGISTRY[f.type](f)

            elif isinstance(f.type, typing._GenericAlias) and f.type._name in [
                'List',
                'Iterable',
            ]:
                sub_type = f.type.__args__[0]
                if sub_type in _TYPES_REGISTRY:
                    decorated_cls.__dataclass_fields__[key] = _TYPES_REGISTRY[sub_type](
                        f
                    )

        return decorated_cls

    if cls is None:
        return wrap

    return wrap(cls)


def is_dataclass(obj) -> bool:
    """Returns True if obj is an instance of :meth:`.dataclass`."""
    return _is_dataclass(obj) and hasattr(obj, 'from_document')


@classmethod
def from_document(cls: 'T', doc: 'Document'):
    if not doc.is_multimodal:
        raise ValueError(
            f'{doc} is not a multimodal doc instantiated from a class wrapped by `docarray.dataclasses.tdataclass`.'
        )

    attributes = {}
    for key, attribute_info in doc._metadata['multi_modal_schema'].items():
        field = cls.__dataclass_fields__[key]
        position = doc._metadata['multi_modal_schema'][key].get('position')
        if (
            attribute_info['type'] == 'bytes'
            and attribute_info['attribute_type'] == AttributeType.PRIMITIVE
        ):
            attributes[key] = base64.b64decode(doc.tags[key].encode())
        elif attribute_info['attribute_type'] in [
            AttributeType.PRIMITIVE,
            AttributeType.ITERABLE_PRIMITIVE,
        ]:
            attributes[key] = doc.tags[key]
        elif attribute_info['attribute_type'] == AttributeType.DOCUMENT:
            attribute_doc = doc.chunks[position]
            attribute = _get_doc_attribute(attribute_doc, field)
            attributes[key] = attribute
        elif attribute_info['attribute_type'] == AttributeType.ITERABLE_DOCUMENT:
            attribute_list = []
            for chunk_doc in doc.chunks[position].chunks:
                attribute_list.append(_get_doc_attribute(chunk_doc, field))
            attributes[key] = attribute_list
        elif attribute_info['attribute_type'] == AttributeType.NESTED:
            nested_cls = field.type
            attributes[key] = _get_doc_nested_attribute(
                doc.chunks[position], nested_cls
            )
        elif attribute_info['attribute_type'] == AttributeType.ITERABLE_NESTED:
            nested_cls = cls.__dataclass_fields__[key].type.__args__[0]
            attribute_list = []
            for chunk_doc in doc.chunks[position].chunks:
                attribute_list.append(_get_doc_nested_attribute(chunk_doc, nested_cls))
            attributes[key] = attribute_list
        else:
            raise AttributeError(f'Invalid attribute at `{key}`')

    return cls(**attributes)


def _get_doc_attribute(attribute_doc: 'Document', field):
    if isinstance(field, Field):
        return field.deserializer(field.name, attribute_doc)
    else:
        raise ValueError('Invalid attribute type')


def _get_doc_nested_attribute(attribute_doc: 'Document', nested_cls):
    if not is_dataclass(nested_cls):
        raise ValueError(f'Nested attribute {nested_cls.__name__} is not a dataclass')
    return nested_cls.from_document(attribute_doc)
