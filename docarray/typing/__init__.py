from typing_extensions import TYPE_CHECKING

from docarray.typing.bytes import ImageBytes
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
    'ImageBytes',
    'ImageTensor',
    'ImageNdArray',
    'ImageBytes',
    'VideoBytes',
    'AudioBytes',
]


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

from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
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

tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor import TensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa

    __all__.extend(
        [
            'TensorFlowTensor',
            'TensorFlowEmbedding',
            'AudioTensorFlowTensor',
            'ImageTensorFlowTensor',
            'VideoTensorFlowTensor',
        ]
    )

#
# def __getattr__(name: str):
#
#     torch_tensors = [
#         'AudioTorchTensor',
#         'TorchEmbedding',
#         'TorchTensor',
#         'VideoTorchTensor',
#         'ImageTorchTensor',
#     ]
#
#     tf_tensors = [
#         'TensorFlowTensor',
#         'TensorFlowEmbedding',
#         'AudioTensorFlowTensor',
#         'ImageTensorFlowTensor',
#         'VideoTensorFlowTensor',
#     ]
#
#     if name in torch_tensors:
#         import_library('torch', raise_error=True)
#
#         from docarray.typing.tensor import TorchEmbedding, TorchTensor  # noqa: F811
#         from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F811
#         from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F811
#         from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F811
#
#         # __all__.extend(torch_tensors)
#
#         if name == 'TorchTensor':
#             __all__.append('TorchTensor')
#             return TorchTensor
#         elif name == 'TorchEmbedding':
#             __all__.append('TorchEmbedding')
#             return TorchEmbedding
#         elif name == 'AudioTorchTensor':
#             __all__.append('AudioTorchTensor')
#             return AudioTorchTensor
#         elif name == 'ImageTorchTensor':
#             __all__.append('ImageTorchTensor')
#             return ImageTorchTensor
#         elif name == 'VideoTorchTensor':
#             __all__.append('VideoTorchTensor')
#             return VideoTorchTensor
#
#     elif name in tf_tensors:
#         import_library('tensorflow', raise_error=True)
#
#         from docarray.typing.tensor import TensorFlowTensor  # noqa
#         from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa
#         from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa
#         from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa
#         from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa
#
#         # __all__.extend(torch_tensors)
#
#         if name == 'TensorFlowTensor':
#             __all__.append('TensorFlowTensor')
#             return TensorFlowTensor
#         elif name == 'TensorFlowEmbedding':
#             __all__.append('TensorFlowEmbedding')
#             return TensorFlowEmbedding
#         elif name == 'AudioTensorFlowTensor':
#             __all__.append('AudioTensorFlowTensor')
#             return AudioTensorFlowTensor
#         elif name == 'ImageTensorFlowTensor':
#             __all__.append('ImageTensorFlowTensor')
#             return ImageTensorFlowTensor
#         elif name == 'VideoTensorFlowTensor':
#             __all__.append('VideoTensorFlowTensor')
#             return VideoTensorFlowTensor
