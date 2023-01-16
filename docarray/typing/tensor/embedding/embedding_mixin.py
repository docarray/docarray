from abc import ABC
from typing import Any, Optional, Tuple, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor


class EmbeddingMixin(AbstractTensor, ABC):
    alternative_type: Optional[Type] = None

    @classmethod
    def _docarray_validate_getitem(cls, item: Any) -> Tuple[int]:
        shape = super()._docarray_validate_getitem(item)
        if len(shape) > 1:
            error_msg = f'`{cls}` can only have a single dimension/axis.'
            if cls.alternative_type:
                error_msg += f' Consider using {cls.alternative_type} instead.'
            raise ValueError(error_msg)
        return shape
