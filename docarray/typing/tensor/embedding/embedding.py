from typing import Union

from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding
from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.embedding.tensorflow import (
        TensorFlowEmbedding as TFEmbedding,
    )


if tf_available and torch_available:
    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding, TFEmbedding]  # type: ignore
elif tf_available:
    AnyEmbedding = Union[NdArrayEmbedding, TFEmbedding]  # type: ignore
elif torch_available:
    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding]  # type: ignore
else:
    AnyEmbedding = Union[NdArrayEmbedding]  # type: ignore

__all__ = ['AnyEmbedding']
