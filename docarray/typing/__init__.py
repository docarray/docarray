from typing_extensions import TYPE_CHECKING

from docarray.typing.bytes import ImageBytes
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
    'NdArrayEmbedding',
    'ImageBytes',
    'ImageTensor',
    'ImageNdArray',
]

torch_tensors = [
    'AudioTorchTensor',
    'TorchEmbedding',
    'TorchTensor',
    'VideoTorchTensor',
    'ImageTorchTensor',
]

tf_tensors = [
    'TensorFlowTensor',
    'TensorFlowEmbedding',
    'AudioTensorFlowTensor',
    'ImageTensorFlowTensor',
    'VideoTensorFlowTensor',
]


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)

        from docarray.typing.tensor import TorchEmbedding, TorchTensor  # noqa: F811
        from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F811
        from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F811
        from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F811

        __all__.extend(torch_tensors)

        if name == 'TorchTensor':
            return TorchTensor
        elif name == 'TorchEmbedding':
            return TorchEmbedding
        elif name == 'AudioTorchTensor':
            return AudioTorchTensor
        elif name == 'ImageTorchTensor':
            return ImageTorchTensor
        elif name == 'VideoTorchTensor':
            return VideoTorchTensor

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor import TensorFlowTensor  # noqa
        from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa
        from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa
        from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa
        from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa

        __all__.extend(torch_tensors)

        if name == 'TensorFlowTensor':
            return TensorFlowTensor
        elif name == 'TensorFlowEmbedding':
            return TensorFlowEmbedding
        elif name == 'AudioTensorFlowTensor':
            return AudioTensorFlowTensor
        elif name == 'ImageTensorFlowTensor':
            return ImageTensorFlowTensor
        elif name == 'VideoTensorFlowTensor':
            return VideoTensorFlowTensor
