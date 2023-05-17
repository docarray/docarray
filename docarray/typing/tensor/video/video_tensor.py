from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch

    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor


tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.video.video_tensorflow_tensor import (
        VideoTensorFlowTensor,
    )

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar("T", bound="VideoTensor")


class VideoTensor:
    """
    Represents a Video tensor object that can be used with TensorFlow, PyTorch, and NumPy type.

    ---

    '''python
    from docarray import BaseDoc
    from docarray.typing import VideoTensor


    class MyDoc(BaseDoc):
        video: VideoTensor


    # Example usage with TensorFlow:
    import tensorflow as tf

    doc = MyDoc(video=tf.zeros(1000, 2))

    # Example usage with PyTorch:
    import torch

    doc = MyDoc(video=torch.zeros(1000, 2))

    # Example usage with NumPy:
    import numpy as np

    doc = MyDoc(video=np.zeros((1000, 2)))
    '''

    Returns:
        Union[VideoTorchTensor, VideoTensorFlowTensor, VideoNdArray]: The validated and converted audio tensor.

    Raises:
        TypeError: If the input value is not a compatible type (torch.Tensor, tensorflow.Tensor, numpy.ndarray).

    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: "ModelField",
        config: "BaseConfig",
    ):
        # Check for TorchTensor first, then TensorFlowTensor, then NdArray
        if torch_available:
            if isinstance(value, TorchTensor):
                return value
            elif isinstance(value, torch.Tensor):
                return VideoTorchTensor._docarray_from_native(value)  # noqa
        if tf_available:
            if isinstance(value, TensorFlowTensor):
                return value
            elif isinstance(value, tf.Tensor):
                return VideoTensorFlowTensor._docarray_from_native(value)  # noqa
        try:
            return VideoNdArray.validate(value, field, config)
        except Exception:  # noqa
            pass
        raise TypeError(
            f"Expected one of [torch.Tensor, tensorflow.Tensor, numpy.ndarray] "
            f"compatible type, got {type(value)}"
        )
