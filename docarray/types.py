from typing import (
    Union,
    TYPE_CHECKING,
    TypeVar,
    Sequence,
    Optional,
    List,
    Dict,
    Generator,
    Iterable,
    Tuple,
    ForwardRef,
)

from dataclasses import dataclass as std_dataclass, is_dataclass

if TYPE_CHECKING:
    import scipy.sparse
    import tensorflow
    import torch
    import numpy as np

    from . import Document

    ArrayType = TypeVar(
        'ArrayType',
        np.ndarray,
        scipy.sparse.spmatrix,
        tensorflow.SparseTensor,
        tensorflow.Tensor,
        torch.Tensor,
        Sequence[float],
    )

    DocumentContentType = Union[bytes, str, ArrayType]
    ProtoValueType = Optional[Union[str, bool, float]]
    StructValueType = Union[
        ProtoValueType, List[ProtoValueType], Dict[str, ProtoValueType]
    ]

    DocumentArraySourceType = Union[
        Sequence[Document], Document, Generator[Document], Iterable[Document]
    ]
    T = TypeVar('T')

    AnyDNN = TypeVar(
        'AnyDNN'
    )  #: The type of any implementation of a Deep Neural Network object

    DocumentArraySingletonIndexType = Union[int, str]
    DocumentArrayMultipleIndexType = Union[
        slice, Sequence[int], Sequence[str], Sequence[bool], Ellipsis
    ]
    DocumentArraySingleAttributeType = Tuple[
        Union[DocumentArraySingletonIndexType, DocumentArrayMultipleIndexType], str
    ]
    DocumentArrayMultipleAttributeType = Tuple[
        Union[DocumentArraySingletonIndexType, DocumentArrayMultipleIndexType],
        Sequence[str],
    ]
    DocumentArrayIndexType = Union[
        DocumentArraySingletonIndexType,
        DocumentArrayMultipleIndexType,
        DocumentArraySingleAttributeType,
        DocumentArrayMultipleAttributeType,
    ]

ImageDocument = TypeVar(
    'ImageDocument',
    ForwardRef('np.ndarray'),
    ForwardRef('tensorflow.Tensor'),
    ForwardRef('torch.Tensor'),
)

BlobDocument = TypeVar(
    'BlobDocument',
    ForwardRef('np.ndarray'),
    bytes,
)

TextDocument = TypeVar('TextDocument', bound=str)


def _get_doc_attribute(attribute_doc: 'Document', attribute_type: str):
    if 'ImageDocument' in attribute_type:
        attribute = attribute_doc.tensor
    elif 'BlobDocument' in attribute_type:
        attribute = attribute_doc.blob
    elif 'TextDocument' in attribute_type:
        attribute = attribute_doc.text
    else:
        raise ValueError('Invalid attribute type')

    return attribute


def _get_doc_nested_attribute(attribute_doc: 'Document', nested_cls):
    if not is_dataclass(nested_cls):
        raise ValueError(f'Nested attribute {nested_cls.__name__} is not a dataclass')
    return nested_cls.from_document(attribute_doc)


@classmethod
def from_document(cls: 'T', doc: 'Document'):
    from .document.mixins.multimodal import AttributeType

    if 'multi_modal_schema' not in doc._metadata:
        raise ValueError('the Document does not correspond to a Multi Modal Document')

    attributes = {}
    for key, field in doc._metadata['multi_modal_schema'].items():
        position = doc._metadata['multi_modal_schema'][key].get('position')
        if field['attribute_type'] in [
            AttributeType.PRIMITIVE,
            AttributeType.ITERABLE_PRIMITIVE,
        ]:
            attributes[key] = doc.tags[key]
        elif field['attribute_type'] == AttributeType.DOCUMENT:
            attribute_doc = doc.chunks[position]
            attribute = _get_doc_attribute(attribute_doc, field['type'])
            attributes[key] = attribute
        elif field['attribute_type'] == AttributeType.ITERABLE_DOCUMENT:
            attribute_list = []
            for chunk_doc in doc.chunks[position].chunks:
                attribute_list.append(_get_doc_attribute(chunk_doc, field['type']))
            attributes[key] = attribute_list
        elif field['attribute_type'] == AttributeType.NESTED:
            nested_cls = cls.__dataclass_fields__[key].type
            attributes[key] = _get_doc_nested_attribute(
                doc.chunks[position], nested_cls
            )
        elif field['attribute_type'] == AttributeType.ITERABLE_NESTED:
            nested_cls = cls.__dataclass_fields__[key].type.__args__[0]
            attribute_list = []
            for chunk_doc in doc.chunks[position].chunks:
                attribute_list.append(_get_doc_nested_attribute(chunk_doc, nested_cls))
            attributes[key] = attribute_list
        else:
            raise ValueError(f'Invalid attribute {key}')

    return cls(**attributes)


def dataclass(cls=None):
    """Extends python standard dataclass decorator to add functionalities to enable multi modality support to Document.
    Returns the same class as was passed in, with dunder methods and from_document method.


    If init is true, an __init__() method is added to the class. If
    repr is true, a __repr__() method is added. If order is true, rich
    comparison dunder methods are added. If unsafe_hash is true, a
    __hash__() method function is added. If frozen is true, fields may
    not be assigned to after instance creation.
    """

    def wrap(cls):
        decorated_cls = std_dataclass(
            cls,
            init=True,
            repr=True,
            eq=True,
            order=False,
            unsafe_hash=False,
            frozen=False,
        )
        setattr(decorated_cls, from_document.__func__.__name__, from_document)
        return decorated_cls

    return wrap(cls)
