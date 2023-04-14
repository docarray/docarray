from typing import Any  # noqa: F401

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.torch_tensor import TorchTensor

torch_base = type(TorchTensor)  # type: Any
embedding_base = type(EmbeddingMixin)  # type: Any


class metaTorchAndEmbedding(torch_base, embedding_base):
    pass


@_register_proto(proto_type_name='torch_embedding')
class TorchEmbedding(TorchTensor, EmbeddingMixin, metaclass=metaTorchAndEmbedding):
    alternative_type = TorchTensor

    def new_empty(self, *args, **kwargs):
        """
        This method enables the deepcopy of `TorchEmbedding` by returning another instance of this subclass.
        If this function is not implemented, the deepcopy will throw an RuntimeError from Torch.
        """
        return self.__class__(TorchTensor.new_empty(self, *args, **kwargs))
