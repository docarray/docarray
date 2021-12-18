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
)

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

    DocumentArrayIndexType = Union[int, str, slice, Sequence[int], Sequence[str], Sequence[bool], Ellipsis]