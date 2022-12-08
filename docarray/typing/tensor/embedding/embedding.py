from typing import TYPE_CHECKING, Union

from docarray.typing.tensor.embedding.ndarray import NdArrayEmbedding

if TYPE_CHECKING:
    from docarray.typing.tensor.embedding.torch import TorchEmbedding

Embedding = Union[NdArrayEmbedding, 'TorchEmbedding']
