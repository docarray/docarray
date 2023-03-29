from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.audio import AudioNdArray
from docarray.typing.tensor.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.image import ImageNdArray, ImageTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor
from docarray.typing.tensor.video import VideoNdArray
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401
    from docarray.typing.tensor.video import VideoTorchTensor  # noqa: F401

__all__ = [
    'NdArray',
    'AnyTensor',
    'AnyEmbedding',
    'NdArrayEmbedding',
    'ImageNdArray',
    'ImageTensor',
    'AudioNdArray',
    'VideoNdArray',
]


def __getattr__(name: str):

    if name not in __all__:
        __all__.append(name)
    torch_tensors = ['TorchEmbedding', 'ImageTorchTensor', 'TorchTensor']
    tf_tensors = ['TensorFlowEmbedding', 'TensorFlowTensor', 'ImageTensorFlowTensor']

    if name in torch_tensors:
        import_library('torch', raise_error=True)

        if name == 'TorchTensor':
            from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa

            return TorchTensor
        elif name == 'TorchEmbedding':
            from docarray.typing.tensor.embedding import TorchEmbedding  # noqa

            return TorchEmbedding
        elif name == 'ImageTorchTensor':
            from docarray.typing.tensor.image import ImageTorchTensor  # noqa

            return ImageTorchTensor

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        if name == 'TensorFlowTensor':
            from docarray.typing.tensor.tensorflow_tensor import (  # noqa
                TensorFlowTensor,
            )

            return TensorFlowTensor
        elif name == 'TensorFlowEmbedding':
            from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa

            return TensorFlowEmbedding
        elif name == 'ImageTensorFlowTensor':
            from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa

            return ImageTensorFlowTensor
