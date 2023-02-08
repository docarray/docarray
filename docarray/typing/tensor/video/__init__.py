from docarray.typing.tensor.video.video_ndarray import VideoNdArray

__all__ = ['VideoNdArray']

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

    __all__.extend(['VideoTorchTensor'])

try:
    import tensorflow as tf  # noqa: F401
except (ImportError, TypeError):
    pass
else:
    from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa: F401
        VideoTensorFlowTensor,
    )

    __all__.extend(['VideoTensorFlowTensor'])
