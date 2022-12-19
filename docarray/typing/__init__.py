from docarray.typing.id import ID
from docarray.typing.tensor import AnyTensor, NdArray
from docarray.typing.tensor.embedding import Embedding
from docarray.typing.url import AnyUrl, ImageUrl, Mesh3DUrl, PointCloud3DUrl, TextUrl

__all__ = [
    'NdArray',
    'Embedding',
    'ImageUrl',
    'TextUrl',
    'Mesh3DUrl',
    'PointCloud3DUrl',
    'AnyUrl',
    'ID',
    'AnyTensor',
]

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor import TorchEmbedding, TorchTensor  # noqa: F401

    __all__.extend(['TorchEmbedding', 'TorchTensor'])
