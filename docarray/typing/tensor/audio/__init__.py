import types
from typing import TYPE_CHECKING

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_tensor import AudioTensor
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.typing.tensor.audio.audio_jax_array import AudioJaxArray  # noqa
    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
        AudioTensorFlowTensor,
    )
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa

__all__ = ['AudioNdArray', 'AudioTensor', 'AudioJaxArray']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'AudioTorchTensor':
        import_library('torch', raise_error=True)
        import docarray.typing.tensor.audio.audio_torch_tensor as lib
    elif name == 'AudioTensorFlowTensor':
        import_library('tensorflow', raise_error=True)
        import docarray.typing.tensor.audio.audio_tensorflow_tensor as lib
    elif name == 'AudioJaxArray':
        import_library('jax', raise_error=True)
        import docarray.typing.tensor.audio.audio_jax_array as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
