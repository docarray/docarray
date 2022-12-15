from docarray.typing.id import ID
from docarray.typing.tensor import NdArray, Tensor
from docarray.typing.tensor.embedding import Embedding
from docarray.typing.url import (
    AnyUrl,
    AudioUrl,
    ImageUrl,
    Mesh3DUrl,
    PointCloud3DUrl,
    TextUrl,
)

__all__ = [
    'NdArray',
    'Embedding',
    'ImageUrl',
    'AudioUrl',
    'TextUrl',
    'Mesh3DUrl',
    'PointCloud3DUrl',
    'AnyUrl',
    'ID',
    'Tensor',
]

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor import TorchEmbedding, TorchTensor  # noqa: F401

    __all__.extend(['TorchEmbedding', 'TorchTensor'])
