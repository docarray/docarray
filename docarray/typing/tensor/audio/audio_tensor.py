from typing import Union

from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray

try:
    import torch  # noqa: F401
except ImportError:
    AudioTensor = AudioNdArray

else:
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor

    AudioTensor = Union[AudioNdArray, AudioTorchTensor]  # type: ignore
