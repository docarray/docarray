from docarray.typing.tensor.embedding.embedding import AnyEmbedding
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

__all__ = ['NdArrayEmbedding', 'AnyEmbedding']

from docarray.utils._internal.misc import import_library

torch_tensors = ['TorchEmbedding']
tf_tensors = ['TensorFlowEmbedding']


def __getattr__(name: str):
    if name in torch_tensors:
        import_library('torch', raise_error=True)
        from docarray.typing.tensor.embedding.torch import TorchEmbedding  # noqa

        __all__.extend(torch_tensors)

    elif name in tf_tensors:
        import_library('tensorflow', raise_error=True)

        from docarray.typing.tensor.embedding.tensorflow import (  # noqa
            TensorFlowEmbedding,
        )

        __all__.extend(tf_tensors)
