from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.typing.tensor.video.video_tensor import VideoTensor

__all__ = ['VideoNdArray', 'VideoTensor']

from docarray.utils._internal.misc import import_library

torch_tensors = ['VideoTorchTensor']
tf_tensors = ['VideoTensorFlowTensor']


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.video.video_torch_tensor import (  # noqa
            VideoTorchTensor,
        )

        __all__.extend(torch_tensors)

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa
            VideoTensorFlowTensor,
        )

        __all__.extend(tf_tensors)
