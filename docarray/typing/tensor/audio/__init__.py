from typing import TYPE_CHECKING

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_tensor import AudioTensor
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
        AudioTensorFlowTensor,
    )
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa

__all__ = ['AudioNdArray', 'AudioTensor']


def __getattr__(name: str):
    if name not in __all__:
        __all__.append(name)

    if name == 'AudioTorchTensor':
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.audio.audio_torch_tensor import (  # noqa
            AudioTorchTensor,
        )

        return AudioTorchTensor

    elif name == 'AudioTensorFlowTensor':
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
            AudioTensorFlowTensor,
        )

        return AudioTensorFlowTensor
