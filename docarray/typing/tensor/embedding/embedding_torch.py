from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.torch_tensor import TorchTensor

torch_base = type(TorchTensor)  # type: Any
embedding_base = type(EmbeddingMixin)  # type: Any


class metaTorchAndEmbedding(torch_base, embedding_base):
    pass


class TorchEmbedding(TorchTensor, EmbeddingMixin, metaclass=metaTorchAndEmbedding):
    alternative_type = TorchTensor
