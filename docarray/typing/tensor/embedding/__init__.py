from docarray.typing.tensor.embedding.embedding import Embedding
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

__all__ = ['NdArrayEmbedding', 'Embedding']

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding  # noqa F401

    __all__.append('TorchEmbedding')
