from typing import Any  # noqa: F401

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor

tensorflow_base = type(TensorFlowTensor)  # type: Any
embedding_base = type(EmbeddingMixin)  # type: Any


class metaTensorFlowAndEmbedding(tensorflow_base, embedding_base):
    pass


@_register_proto(proto_type_name='tensorflow_embedding')
class TensorFlowEmbedding(
    TensorFlowTensor, EmbeddingMixin, metaclass=metaTensorFlowAndEmbedding
):
    alternative_type = TensorFlowTensor
