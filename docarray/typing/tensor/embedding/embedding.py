from typing import Dict, Generic, Type, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding
from docarray.typing.tensor.tensor import NdArray
from docarray.utils.misc import is_tf_available, is_torch_available

T = TypeVar('T', bound=AbstractTensor)


class EmbeddingTensor(NdArrayEmbedding, Generic[T]):
    _EMBEDDING_TENSOR: Dict[Type[AbstractTensor], Type] = {}

    @classmethod
    def __class_getitem__(cls, item: Union[Type[AbstractTensor], TypeVar]):  # type: ignore
        if not isinstance(item, type):
            return Generic.__class_getitem__.__func__(cls, item)  # type: ignore
        return cls._EMBEDDING_TENSOR[item]


EmbeddingTensor._EMBEDDING_TENSOR[NdArray] = NdArrayEmbedding

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding
    from docarray.typing.tensor.tensor import TorchTensor

    EmbeddingTensor._EMBEDDING_TENSOR[TorchTensor] = TorchEmbedding


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.embedding.tensorflow import (
        TensorFlowEmbedding as TFEmbedding,
    )
    from docarray.typing.tensor.tensor import TensorFlowTensor

    EmbeddingTensor._EMBEDDING_TENSOR[TensorFlowTensor] = TFEmbedding


if tf_available and torch_available:
    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding, TFEmbedding]  # type: ignore
elif tf_available:
    AnyEmbedding = Union[NdArrayEmbedding, TFEmbedding]  # type: ignore
elif torch_available:
    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding]  # type: ignore
else:
    AnyEmbedding = Union[NdArrayEmbedding]  # type: ignore

__all__ = ['AnyEmbedding']
