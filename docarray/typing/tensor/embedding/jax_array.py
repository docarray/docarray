from typing import Any  # noqa: F401

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.jaxarray import JaxArray

jax_base = type(JaxArray)  # type: Any
embedding_base = type(EmbeddingMixin)  # type: Any


class metaJaxAndEmbedding(jax_base, embedding_base):
    pass


@_register_proto(proto_type_name='jaxarray_embedding')
class JaxArrayEmbedding(JaxArray, EmbeddingMixin, metaclass=metaJaxAndEmbedding):
    alternative_type = JaxArray
