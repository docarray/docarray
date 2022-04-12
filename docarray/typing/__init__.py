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

if TYPE_CHECKING:
    import scipy.sparse
    import tensorflow
    import torch
    import numpy as np
    from PIL.Image import Image as PILImage

    from .. import Document

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


Image = TypeVar(
    'Image',
    str,
    ForwardRef('np.ndarray'),
    ForwardRef('PILImage'),
)
Text = TypeVar('Text', bound=str)
Audio = TypeVar('Audio', str, ForwardRef('np.ndarray'))
Video = TypeVar('Video', str, ForwardRef('np.ndarray'))
Mesh = TypeVar('Mesh', str, ForwardRef('np.ndarray'))
Tabular = TypeVar('Tabular', bound=str)
Blob = TypeVar('Blob', str, bytes)
JSON = TypeVar('JSON', str, dict)
