from docarray.typing.tensor.video.video_ndarray import VideoNdArray

__all__ = ['VideoNdArray']

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

    __all__.extend(['VideoTorchTensor'])
