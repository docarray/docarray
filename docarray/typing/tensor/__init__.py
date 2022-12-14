from typing import Dict, Set, Type

from docarray.typing.tensor.embedding import Embedding, NdArrayEmbedding
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.tensor import Tensor

__all__ = [
    'NdArray',
    'Tensor',
    'Embedding',
    'NdArrayEmbedding',
    'framework_types',
    'type_to_framework',
]

framework_types: Dict[str, Set] = {'numpy': {NdArray, NdArrayEmbedding}, 'torch': set()}

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.embedding import TorchEmbedding  # noqa: F401
    from docarray.typing.tensor.torch_tensor import TorchTensor  # noqa: F401

    __all__.extend(['TorchEmbedding', 'TorchTensor'])
    framework_types['torch'] = {TorchTensor, TorchEmbedding}

type_to_framework: Dict[Type, str] = {
    type_: framework
    for framework, type_set in framework_types.items()
    for type_ in type_set
}
