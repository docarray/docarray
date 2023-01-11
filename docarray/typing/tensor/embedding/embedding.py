from typing import Union

from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

try:
    import torch  # noqa: F401
except ImportError:
    AnyEmbedding = Union[NdArrayEmbedding]  # type: ignore

else:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding  # noqa: F401

    AnyEmbedding = Union[NdArrayEmbedding, TorchEmbedding]  # type: ignore

__all__ = ['AnyEmbedding']
