from abc import ABC
from typing import Any, Tuple, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.torch_tensor import TorchTensor

T = TypeVar('T', bound='Embedding')


class EmbeddingMixin(AbstractTensor, ABC):
    alternative_type = None

    @classmethod
    def __validate_getitem__(cls, item: Any) -> Tuple[int]:
        shape = super().__validate_getitem__(item)
        if len(shape) > 1:
            error_msg = f'`{cls}` can only have a single dimension/axis.'
            if cls.alternative_type:
                error_msg += f' Consider using {cls.alternative_type} instead.'
            raise ValueError(error_msg)
        return shape


class NdArrayEmbedding(NdArray, EmbeddingMixin):
    alternative_type = NdArray


class TorchEmbedding(TorchTensor, EmbeddingMixin):
    alternative_type = TorchTensor


Embedding = Union[TorchEmbedding, NdArrayEmbedding]
