from typing_extensions import TYPE_CHECKING

from docarray.typing.tensor.embedding.embedding import AnyEmbedding
from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.typing.tensor.embedding.tensorflow import TensorFlowEmbedding  # noqa
    from docarray.typing.tensor.embedding.torch import TorchEmbedding  # noqa

__all__ = ['NdArrayEmbedding', 'AnyEmbedding']


def __getattr__(name: str):
    if name == 'TorchEmbedding':
        import_library('torch', raise_error=True)
        import docarray.typing.tensor.embedding.torch as lib
    elif name == 'TensorFlowEmbedding':
        import_library('tensorflow', raise_error=True)
        import docarray.typing.tensor.embedding.tensorflow as lib

    tensor_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return tensor_cls
