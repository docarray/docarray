from docarray.typing.bytes import ImageBytes
from docarray.typing.id import ID
from docarray.typing.tensor import ImageNdArray, ImageTensor
from docarray.typing.tensor.audio import AudioNdArray
from docarray.typing.tensor.embedding.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor
from docarray.typing.tensor.video import VideoNdArray
from docarray.typing.url import (
    AnyUrl,
    AudioUrl,
    ImageUrl,
    Mesh3DUrl,
    PointCloud3DUrl,
    TextUrl,
    VideoUrl,
)

__all__ = [
    'NdArray',
    'NdArrayEmbedding',
    'AudioNdArray',
    'VideoNdArray',
    'AnyEmbedding',
    'ImageUrl',
    'AudioUrl',
    'TextUrl',
    'Mesh3DUrl',
    'PointCloud3DUrl',
    'VideoUrl',
    'AnyUrl',
    'ID',
    'AnyTensor',
    'TensorFlowTensor',
    'NdArrayEmbedding',
    'ImageBytes',
    'ImageTensor',
    'ImageNdArray',
]

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor import TorchEmbedding, TorchTensor  # noqa: F401
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa:  F401
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

    __all__.extend(
        [
            'AudioTorchTensor',
            'TorchEmbedding',
            'TorchTensor',
            'VideoTorchTensor',
            'ImageTorchTensor',
        ]
    )

try:
    import tensorflow as tf  # type: ignore # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor import TensorFlowTensor  # noqa: F401

    __all__.extend(['TensorFlowTensor'])
