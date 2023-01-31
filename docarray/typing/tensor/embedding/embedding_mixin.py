from abc import ABC
from typing import Optional, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor


class EmbeddingMixin(AbstractTensor, ABC):
    alternative_type: Optional[Type] = None

    def __class_getitem__(cls, item):
        shape, _ = cls._parse_item(item)
        if shape and len(shape.split()) > 1:
            error_msg = f'`{cls}` can only have a single dimension/axis.'
            if cls.alternative_type:
                error_msg += f' Consider using {cls.alternative_type} instead.'
            raise ValueError(error_msg)
        return super().__class_getitem__(item)
