from typing import Union

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray

try:
    torch_available = True
    import torch  # noqa: F401

    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor
except ImportError:
    torch_available = False

try:
    tf_available = True
    import tensorflow as tf  # noqa: F401

    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (
        AudioTensorFlowTensor as AudioTFTensor,
    )
except ImportError:
    tf_available = False

if tf_available and torch_available:
    AudioTensor = Union[AudioNdArray, AudioTorchTensor, AudioTFTensor]  # type: ignore
elif tf_available:
    AudioTensor = Union[AudioNdArray, AudioTFTensor]  # type: ignore
elif tf_available:
    AudioTensor = Union[AudioNdArray, AudioTorchTensor]  # type: ignore
else:
    AudioTensor = Union[AudioNdArray]  # type: ignore
