from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.video.abstract_video_tensor import AbstractVideoTensor

T = TypeVar('T', bound='VideoNdArray')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


class VideoNdArray(AbstractVideoTensor, NdArray):
    """
    Subclass of NdArray, to represent a video tensor.
    Adds video-specific features to the tensor.

    EXAMPLE USAGE

    """

    _PROTO_FIELD_NAME = 'video_ndarray'

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        array = super().validate(value=value, field=field, config=config)
        if array.ndim not in [3, 4] or array.shape[-1] != 3:
            raise ValueError(
                f'Expects tensor with 3 or 4 dimensions and the last dimension equal'
                f' to 3, but received {array.shape} in {array.dtype}'
            )
        else:
            return array
