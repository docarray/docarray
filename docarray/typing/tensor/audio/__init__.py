from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray

__all__ = ['AudioNdArray']

from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch  # noqa: F401

    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa

    __all__.extend(['AudioTorchTensor'])


tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # noqa: F401

    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (  # noqa
        AudioTensorFlowTensor,
    )

    __all__.extend(['AudioTensorFlowTensor'])
