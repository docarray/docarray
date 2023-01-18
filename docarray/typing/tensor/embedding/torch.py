from typing import Any  # noqa: F401

from docarray.typing.proto_register import register_proto
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.torch_tensor import TorchTensor

torch_base = type(TorchTensor)  # type: Any
embedding_base = type(EmbeddingMixin)  # type: Any


class metaTorchAndEmbedding(torch_base, embedding_base):
    pass


@register_proto(proto_type_name='torch_embedding')
class TorchEmbedding(TorchTensor, EmbeddingMixin, metaclass=metaTorchAndEmbedding):
    alternative_type = TorchTensor
