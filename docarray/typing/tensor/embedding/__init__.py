from docarray.typing.tensor.embedding.embedding import AnyEmbedding
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

__all__ = ['NdArrayEmbedding', 'AnyEmbedding']

from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding  # noqa F401

    __all__.append('TorchEmbedding')


tf_available = is_tf_available()
if tf_available:
    from docarray.typing.tensor.embedding.tensorflow import (  # noqa F401
        TensorFlowEmbedding,
    )

    __all__.append('TensorFlowEmbedding')
