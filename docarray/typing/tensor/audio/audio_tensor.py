from typing import Union

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor

tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.audio.audio_tensorflow_tensor import (
        AudioTensorFlowTensor as AudioTFTensor,
    )


AudioTensor = AudioNdArray
if tf_available and torch_available:
    AudioTensor = Union[AudioNdArray, AudioTorchTensor, AudioTFTensor]  # type: ignore
elif tf_available:
    AudioTensor = Union[AudioNdArray, AudioTFTensor]  # type: ignore
elif torch_available:
    AudioTensor = Union[AudioNdArray, AudioTorchTensor]  # type: ignore
