from typing import Union

from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

try:
    torch_available = True
    import torch  # noqa: F401

    from docarray.typing.tensor.embedding.torch import TorchEmbedding
except ImportError:
    torch_available = False

try:
    tf_available = True
    import tensorflow as tf  # noqa: F401

    from docarray.typing.tensor.embedding.tensorflow import (
        TensorFlowEmbedding as TFEmbedding,
    )
except ImportError:
    tf_available = False

if tf_available and torch_available:
    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding, TFEmbedding]  # type: ignore
elif tf_available:
    AnyEmbedding = Union[NdArrayEmbedding, TFEmbedding]  # type: ignore
elif tf_available:
    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding]  # type: ignore
else:
    AnyEmbedding = Union[NdArrayEmbedding]  # type: ignore

__all__ = ['AnyEmbedding']
