from docarray.typing.id import ID
from docarray.typing.tensor.audio import AudioNdArray
from docarray.typing.tensor.embedding.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.url import (
    AnyUrl,
    AudioUrl,
    ImageUrl,
    Mesh3DUrl,
    PointCloud3DUrl,
    TextUrl,
)

__all__ = [
    'AudioNdArray',
    'NdArray',
    'NdArrayEmbedding',
    'AnyEmbedding',
    'ImageUrl',
    'AudioUrl',
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
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa

    __all__.extend(['AudioTorchTensor', 'TorchEmbedding', 'TorchTensor'])
