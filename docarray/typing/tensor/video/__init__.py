from docarray.typing.tensor.video.video_ndarray import VideoNdArray

__all__ = ['VideoNdArray']

from docarray.utils._internal.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

    __all__.extend(['VideoTorchTensor'])


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa: F401
        VideoTensorFlowTensor,
    )

    __all__.extend(['VideoTensorFlowTensor'])
