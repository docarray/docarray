try:
    import torch  # noqa: F401

    from docarray.typing.tensor.torch_tensor import TorchEmbedding  # noqa: F401
except ImportError:
    pass

from docarray.typing.tensor.embedding.embedding import Embedding
from docarray.typing.tensor.embedding.embedding_ndarray import NdArrayEmbedding

__all__ = ['NdArrayEmbedding', 'Embedding']
