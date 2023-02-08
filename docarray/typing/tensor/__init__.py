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
    'TensorFlowTensor',
]

from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch  # noqa: F401

    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    __all__.extend(['TorchEmbedding', 'TorchTensor', 'ImageTorchTensor'])

    torch_available = is_torch_available()


tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore # noqa: F401

    from docarray.typing.tensor.embedding import TensorFlowEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTensorFlowTensor  # noqa: F401
    from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor  # noqa: F401

    __all__.extend(['TensorFlowEmbedding', 'TensorFlowTensor', 'ImageTensorFlowTensor'])
