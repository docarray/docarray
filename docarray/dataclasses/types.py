import base64
import copy
import functools
import typing
from dataclasses import (
    dataclass as _dataclass,
    Field as _Field,
    is_dataclass as _is_dataclass,
    field as _field,
    MISSING,
)
from enum import Enum
from pathlib import Path
from typing import (
    TypeVar,
    ForwardRef,
    Callable,
    Optional,
    TYPE_CHECKING,
    overload,
    Dict,
    Type,
)

from .setter import (
    image_setter,
    text_setter,
    audio_setter,
    json_setter,
)
from .getter import (
    image_getter,
    text_getter,
    audio_getter,
    json_getter,
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
        *,
        setter: Callable,
        getter: Callable,
        _source_field: Optional[_Field] = None,
        **kwargs,
    ):
        self.copy_from(_source_field if _source_field else _field(**kwargs))
        self.setter = setter
        self.getter = getter

    def copy_from(self, f: '_Field'):
        for s in f.__slots__:
            setattr(self, s, getattr(f, s))

    def get_field(self, doc: 'Document'):
        return self.getter(doc, self.name)

    def set_field(self, val) -> 'Document':
        return self.setter(self.name, val)


@overload
def field(
    *,
    _source_field: Optional[_Field] = None,  # Privately used
    setter: Callable,
    getter: Callable,
    default=MISSING,
    default_factory=MISSING,
    init=True,
    repr=True,
    hash=None,
    compare=True,
    metadata=None,
) -> _Field:
    ...


def field(**kwargs) -> Field:
    return Field(**kwargs)


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
    Image: lambda x: field(setter=image_setter, getter=image_getter, _source_field=x),
    Text: lambda x: field(setter=text_setter, getter=text_getter, _source_field=x),
    Audio: lambda x: field(setter=audio_setter, getter=audio_getter, _source_field=x),
    JSON: lambda x: field(setter=json_setter, getter=json_getter, _source_field=x),
}


@overload
def dataclass(
    cls: Optional['T'] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    type_var_map: Optional[Dict[TypeVar, Callable[['_Field'], 'Field']]] = None,
) -> 'T':
    ...


def dataclass(
    cls: Optional['T'] = None,
    *,
    type_var_map: Optional[Dict[TypeVar, Callable[['_Field'], 'Field']]] = None,
    **kwargs,
) -> 'T':
    """Annotates a class as a DocArray dataclass type.

    Example usage:

    >>> from docarray import dataclass, Image, Text
    >>>
    >>> @dataclass:
    >>> class X:
    >>>     banner: Image = 'apple.png'
    >>>     description: Text = 'This is a big red apple.'

    :param type_var_map: a mapping from TypeVar to a callable that gives Field.

        .. highlight:: python
        .. code-block:: python

            _TYPES_REGISTRY = {
                Image: lambda x: field(setter=image_setter, getter=image_getter, _source_field=x),
                Text: lambda x: field(setter=text_setter, getter=text_getter, _source_field=x),
            }

        The default mapping will be overrided by this new mapping if they collide on the keys.

    """

    if not type_var_map:
        type_var_map = _TYPES_REGISTRY
    else:
        r = copy.deepcopy(_TYPES_REGISTRY)
        r.update(type_var_map)
        type_var_map = r

    from docarray import Document

    def deco(f):
        """
        Set Decorator function.

        :param f: function the decorator is used for
        :return: wrapper
        """

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if not kwargs and len(args) == 2 and isinstance(args[1], Document):
                return f(args[0], **from_document(type(args[0]), args[1]))
            else:
                return f(*args, **kwargs)

        return wrapper

    def wrap(cls):
        decorated_cls = _dataclass(cls, **kwargs)

        # inject flag for recognizing this is a multimodal dataclass
        setattr(decorated_cls, '__is_multimodal__', True)

        # wrap init so `MMDoc(document)` is possible
        if getattr(decorated_cls, '__init__'):
            decorated_cls.__init__ = deco(decorated_cls.__init__)

        for key, f in decorated_cls.__dataclass_fields__.items():
            if isinstance(f, Field):
                continue

            if f.type in type_var_map:
                decorated_cls.__dataclass_fields__[key] = type_var_map[f.type](f)

            elif isinstance(f.type, typing._GenericAlias) and f.type._name in [
                'List',
                'Iterable',
            ]:
                sub_type = f.type.__args__[0]
                if sub_type in type_var_map:
                    decorated_cls.__dataclass_fields__[key] = type_var_map[sub_type](f)

        return decorated_cls

    if cls is None:
        return wrap

    return wrap(cls)


def is_multimodal(obj) -> bool:
    """Returns True if obj is an instance of :meth:`.dataclass`."""
    return _is_dataclass(obj) and hasattr(obj, '__is_multimodal__')


def from_document(cls: Type['T'], doc: 'Document') -> 'T':
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

    return attributes


def _get_doc_attribute(attribute_doc: 'Document', field):
    if isinstance(field, Field):
        return field.get_field(attribute_doc)
    else:
        raise ValueError('Invalid attribute type')


def _get_doc_nested_attribute(attribute_doc: 'Document', nested_cls: Type['T']) -> 'T':
    if not is_multimodal(nested_cls):
        raise ValueError(f'Nested attribute {nested_cls.__name__} is not a dataclass')
    return nested_cls(**from_document(nested_cls, attribute_doc))
