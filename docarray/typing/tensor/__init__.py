from docarray.typing.tensor.embedding import (  # , TorchEmbedding
    Embedding,
    NdArrayEmbedding,
)
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import Tensor

# from docarray.typing.tensor.torch_tensor import TorchTensor

__all__ = [
    'NdArray',
    # 'TorchTensor',
    'Tensor',
    'Embedding',
    # 'TorchEmbedding',
    'NdArrayEmbedding',
]

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    __all__.extend(['TorchEmbedding', 'TorchTensor'])
