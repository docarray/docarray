from typing import TYPE_CHECKING

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_tensor import AudioTensor

__all__ = ['AudioNdArray', 'AudioTensor']

from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
        AudioTensorFlowTensor,
    )
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa


def __getattr__(name: str):
    if name == 'AudioTorchTensor':
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.audio.audio_torch_tensor import (  # noqa
            AudioTorchTensor,
        )

        __all__.append('AudioTorchTensor')
        return AudioTorchTensor

    elif name == 'AudioTensorFlowTensor':
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
            AudioTensorFlowTensor,
        )

        __all__.append('AudioTensorFlowTensor')
        return AudioTensorFlowTensor
