from typing import Union

from docarray.typing.tensor.video.video_ndarray import VideoNdArray

try:
    import torch  # noqa: F401
except ImportError:
    VideoTensor = VideoNdArray

else:
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor

    VideoTensor = Union[VideoNdArray, VideoTorchTensor]  # type: ignore
