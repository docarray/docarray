try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.embedding.embedding_torch import TorchEmbedding  # type: ignore # noqa: [F401,E510]


from docarray.typing.tensor.embedding.embedding import Embedding
from docarray.typing.tensor.embedding.embedding_ndarray import NdArrayEmbedding

__all__ = ['NdArrayEmbedding', 'Embedding']
