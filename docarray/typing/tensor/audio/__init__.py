from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.audio.audio_tensor import AudioTensor

__all__ = ['AudioNdArray', 'AudioTensor']

from docarray.utils._internal.misc import import_library

torch_tensors = ['AudioTorchTensor']
tf_tensors = ['AudioTensorFlowTensor']


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.audio.audio_torch_tensor import (  # noqa
            AudioTorchTensor,
        )

        __all__.extend(['AudioTorchTensor'])

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
            AudioTensorFlowTensor,
        )

        __all__.extend(tf_tensors)
