from docarray.typing.tensor.embedding.embedding import AnyEmbedding
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

__all__ = ['NdArrayEmbedding', 'AnyEmbedding']

try:
    import torch  # noqa: F401
except ImportError:
    pass
else:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding  # noqa F401

    __all__.append('TorchEmbedding')

try:
    import tensorflow as tf  # noqa: F401
except (ImportError, TypeError):
    pass
else:
    from docarray.typing.tensor.embedding.tensorflow import (  # noqa F401
        TensorFlowEmbedding,
    )

    __all__.append('TensorFlowEmbedding')
