from typing_extensions import TYPE_CHECKING

from docarray.typing.bytes import AudioBytes, ImageBytes, VideoBytes
from docarray.typing.id import ID
from docarray.typing.tensor import ImageNdArray, ImageTensor
from docarray.typing.tensor.audio import AudioNdArray
from docarray.typing.tensor.embedding.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
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
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor import TensorFlowTensor  # noqa:  F401
    from docarray.typing.tensor import TorchEmbedding, TorchTensor  # noqa: F401
    from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F401
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F401


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
    'ImageTensor',
    'ImageNdArray',
    'ImageBytes',
    'VideoBytes',
    'AudioBytes',
]


def __getattr__(name: str):
    if 'Torch' in name:
        import_library('torch', raise_error=True)
    elif 'TensorFlow' in name:
        import_library('tensorflow', raise_error=True)

    from docarray.typing import tensor

    T = getattr(tensor, name)
    if name not in __all__:
        __all__.append(name)

    return T

    # if name not in __all__:
    #     __all__.append(name)
    # if 'Torch' in name:
    #     import_library('torch', raise_error=True)
    #
    #     if name == 'TorchTensor':
    #         from docarray.typing.tensor import TorchTensor  # noqa: F811
    #
    #         return TorchTensor
    #     elif name == 'TorchEmbedding':
    #         from docarray.typing.tensor import TorchEmbedding  # noqa: F811
    #
    #         return TorchEmbedding
    #     elif name == 'AudioTorchTensor':
    #         from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F811
    #
    #         return AudioTorchTensor
    #     elif name == 'ImageTorchTensor':
    #         from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F811
    #
    #         return ImageTorchTensor
    #     elif name == 'VideoTorchTensor':
    #         from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F811
    #
    #         return VideoTorchTensor
    # elif 'TensorFlow' in name:
    #     import_library('tensorflow', raise_error=True)
    #     if name == 'TensorFlowTensor':
    #         from docarray.typing.tensor import TensorFlowTensor  # noqa
    #
    #         return TensorFlowTensor
    #     elif name == 'TensorFlowEmbedding':
    #         from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa
    #
    #         return TensorFlowEmbedding
    #     elif name == 'AudioTensorFlowTensor':
    #         from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa
    #
    #         return AudioTensorFlowTensor
    #     elif name == 'ImageTensorFlowTensor':
    #         from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa
    #
    #         return ImageTensorFlowTensor
    #     elif name == 'VideoTensorFlowTensor':
    #         from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa
    #
    #         return VideoTensorFlowTensor
