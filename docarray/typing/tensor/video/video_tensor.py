from typing import Union

from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.video.video_tensorflow_tensor import (
        VideoTensorFlowTensor as VideoTFTensor,
    )

if tf_available and torch_available:
    VideoTensor = Union[VideoNdArray, VideoTorchTensor, VideoTFTensor]  # type: ignore
elif tf_available:
    VideoTensor = Union[VideoNdArray, VideoTFTensor]  # type: ignore
elif torch_available:
    VideoTensor = Union[VideoNdArray, VideoTorchTensor]  # type: ignore
else:
    VideoTensor = Union[VideoNdArray]  # type: ignore
