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

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.image import ImageTorchTensor  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    __all__.extend(['TorchEmbedding', 'TorchTensor', 'ImageTorchTensor'])
