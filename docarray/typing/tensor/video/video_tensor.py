from typing import Union

from docarray.typing.tensor.video.video_ndarray import VideoNdArray

try:
    torch_available = True
    import torch  # noqa: F401

    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor
except ImportError:
    torch_available = False

try:
    tf_available = True
    import tensorflow as tf  # noqa: F401

    from docarray.typing.tensor.video.video_tensorflow_tensor import (
        VideoTensorFlowTensor as VideoTFTensor,
    )
except ImportError:
    tf_available = False

if tf_available and torch_available:
    VideoTensor = Union[VideoNdArray, VideoTorchTensor, VideoTFTensor]  # type: ignore
elif tf_available:
    VideoTensor = Union[VideoNdArray, VideoTFTensor]  # type: ignore
elif tf_available:
    VideoTensor = Union[VideoNdArray, VideoTorchTensor]  # type: ignore
else:
    VideoTensor = Union[VideoNdArray]  # type: ignore
