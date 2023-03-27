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

torch_tensors = ['TorchEmbedding', 'ImageTorchTensor', 'TorchTensor']
tf_tensors = ['TensorFlowEmbedding', 'TensorFlowTensor', 'ImageTensorFlowTensor']


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)

        from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
        from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
        from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

        __all__.extend(torch_tensors)

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
        from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
        from docarray.typing.tensor.tensorflow_tensor import (  # noqa: F401
            TensorFlowTensor,
        )

        __all__.extend(tf_tensors)
