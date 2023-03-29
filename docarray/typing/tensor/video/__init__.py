from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.video.video_ndarray import VideoNdArray
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa
        VideoTensorFlowTensor,
    )
    from docarray.typing.tensor.video.video_torch_tensor import VideoTorchTensor  # noqa

__all__ = ['VideoNdArray', 'VideoTensor']


def __getattr__(name: str):

    if name not in __all__:
        __all__.append(name)

    if name == 'VideoTorchTensor':
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.video.video_torch_tensor import (  # noqa
            VideoTorchTensor,
        )

        return VideoTorchTensor

    elif name == 'VideoTensorFlowTensor':
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.video.video_tensorflow_tensor import (  # noqa
            VideoTensorFlowTensor,
        )

        return VideoTensorFlowTensor
