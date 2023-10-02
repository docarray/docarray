import types

from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.audio import AudioNdArray
from docarray.typing.tensor.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.image import ImageNdArray, ImageTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.tensor.video import VideoNdArray
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor.audio import AudioJaxArray  # noqa: F401
    from docarray.typing.tensor.audio import AudioTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.audio import AudioTorchTensor  # noqa: F401
    from docarray.typing.tensor.embedding import JaxArrayEmbedding  # noqa F401
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageJaxArray  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.jaxarray import JaxArray  # noqa: F401
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoJaxArray  # noqa: F401
    from docarray.typing.tensor.video import VideoTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F401

__all__ = [
    'NdArray',
    'AnyTensor',
    'AnyEmbedding',
    'NdArrayEmbedding',
    'ImageNdArray',
    'ImageTensor',
    'AudioNdArray',
    'VideoNdArray',
]


def __getattr__(name: str):
    if 'Torch' in name:
        import_library('torch', raise_error=True)
    elif 'TensorFlow' in name:
        import_library('tensorflow', raise_error=True)
    elif 'Jax' in name:
        import_library('jax', raise_error=True)

    lib: types.ModuleType
    if name == 'TorchTensor':
        import docarray.typing.tensor.torch_tensor as lib
    elif name == 'TensorFlowTensor':
        import docarray.typing.tensor.tensorflow_tensor as lib
    elif name == 'JaxArray':
        import docarray.typing.tensor.jaxarray as lib
    elif name in ['TorchEmbedding', 'TensorFlowEmbedding', 'JaxArrayEmbedding']:
        import docarray.typing.tensor.embedding as lib
    elif name in ['ImageTorchTensor', 'ImageTensorFlowTensor', 'ImageJaxArray']:
        import docarray.typing.tensor.image as lib
    elif name in ['AudioTorchTensor', 'AudioTensorFlowTensor', 'AudioJaxArray']:
        import docarray.typing.tensor.audio as lib
    elif name in ['VideoTorchTensor', 'VideoTensorFlowTensor', 'VideoJaxArray']:
        import docarray.typing.tensor.video as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
