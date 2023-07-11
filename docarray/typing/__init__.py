from typing_extensions import TYPE_CHECKING

from docarray.typing.bytes import AudioBytes, ImageBytes, VideoBytes
from docarray.typing.id import ID
from docarray.typing.tensor import ImageNdArray, ImageTensor
from docarray.typing.tensor.audio import AudioNdArray, AudioTensor
from docarray.typing.tensor.embedding.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.tensor.video import VideoNdArray, VideoTensor
from docarray.typing.url import (
    AnyUrl,
    AudioUrl,
    ImageUrl,
    Mesh3DUrl,
    PointCloud3DUrl,
    TextUrl,
    VideoUrl,
)
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor import TensorFlowTensor  # noqa:  F401
    from docarray.typing.tensor import (  # noqa: F401
        JaxArray,
        JaxArrayEmbedding,
        TorchEmbedding,
        TorchTensor,
    )
    from docarray.typing.tensor.audio import AudioJaxArray  # noqa: F401
    from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F401
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageJaxArray  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoJaxArray  # noqa: F401
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
    'AudioTensor',
    'VideoTensor',
    'ImageNdArray',
    'ImageBytes',
    'VideoBytes',
    'AudioBytes',
]


_torch_tensors = [
    'TorchTensor',
    'TorchEmbedding',
    'ImageTorchTensor',
    'AudioTorchTensor',
    'VideoTorchTensor',
]
_tf_tensors = [
    'TensorFlowTensor',
    'TensorFlowEmbedding',
    'ImageTensorFlowTensor',
    'AudioTensorFlowTensor',
    'VideoTensorFlowTensor',
]

_jax_tensors = [
    'JaxArray',
    'JaxArrayEmbedding',
    'VideoJaxArray',
    'AudioJaxArray',
    'ImageJaxArray',
]

__all_test__ = __all__ + _torch_tensors


def __getattr__(name: str):
    if name in _torch_tensors:
        import_library('torch', raise_error=True)
    elif name in _tf_tensors:
        import_library('tensorflow', raise_error=True)
    elif name in _jax_tensors:
        import_library('jax', raise_error=True)
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    import docarray.typing.tensor

    tensor_cls = getattr(docarray.typing.tensor, name)
    if name not in __all__:
        __all__.append(name)

    return tensor_cls
