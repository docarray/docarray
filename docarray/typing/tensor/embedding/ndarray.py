from docarray.typing.proto_register import register_proto
from docarray.typing.tensor.embedding.embedding_mixin import EmbeddingMixin
from docarray.typing.tensor.ndarray import NdArray


@register_proto(proto_type_name='ndarray_embedding')
class NdArrayEmbedding(NdArray, EmbeddingMixin):
    alternative_type = NdArray
