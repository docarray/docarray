from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoNdArray')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


class VideoNdArray(NdArray, VideoTensorMixin):
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
        tensor = super().validate(value=value, field=field, config=config)
        return VideoTensorMixin.validate_shape(cls, value=tensor)
