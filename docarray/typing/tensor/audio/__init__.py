from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray

__all__ = ['AudioNdArray']

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.audio.audio_torch_tensor import AudioTorchTensor  # noqa

    __all__.extend(['AudioTorchTensor'])
