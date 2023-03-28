from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.embedding import AnyEmbedding, NdArrayEmbedding
from docarray.typing.tensor.image import ImageNdArray, ImageTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import AnyTensor

__all__ = [
    'NdArray',
    'AnyTensor',
    'AnyEmbedding',
    'NdArrayEmbedding',
    'ImageNdArray',
    'ImageTensor',
]

from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

torch_tensors = ['TorchEmbedding', 'ImageTorchTensor', 'TorchTensor']
tf_tensors = ['TensorFlowEmbedding', 'TensorFlowTensor', 'ImageTensorFlowTensor']


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)

        from docarray.typing.tensor.embedding import TorchEmbedding  # noqa
        from docarray.typing.tensor.image import ImageTorchTensor  # noqa
        from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa

        __all__.extend(torch_tensors)

        if name == 'TorchTensor':
            return TorchTensor
        elif name == 'TorchEmbedding':
            return TorchEmbedding
        elif name == 'ImageTorchTensor':
            return ImageTorchTensor

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa
        from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa
        from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa

        __all__.extend(tf_tensors)

        if name == 'TensorFlowTensor':
            return TensorFlowTensor
        elif name == 'TensorFlowEmbedding':
            return TensorFlowEmbedding
        elif name == 'ImageTensorFlowTensor':
            return ImageTensorFlowTensor
