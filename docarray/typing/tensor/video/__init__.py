import types

from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor.video.video_jax_array import VideoJaxArray  # noqa
    from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa
        VideoTensorFlowTensor,
    )
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

__all__ = ['VideoNdArray', 'VideoTensor']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'VideoTorchTensor':
        import_library('torch', raise_error=True)
        import docarray.typing.tensor.video.video_torch_tensor as lib
    elif name == 'VideoTensorFlowTensor':
        import_library('tensorflow', raise_error=True)
        import docarray.typing.tensor.video.video_tensorflow_tensor as lib
    elif name == 'VideoJaxArray':
        import_library('jax', raise_error=True)
        import docarray.typing.tensor.video.video_jax_array as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
