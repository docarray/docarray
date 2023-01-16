from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoTorchTensor')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


class VideoTorchTensor(TorchTensor, VideoTensorMixin, metaclass=metaTorchAndNode):
    """
    Subclass of TorchTensor, to represent a video tensor.
    Adds video-specific features to the tensor.

    EXAMPLE USAGE

    """

    _PROTO_FIELD_NAME = 'video_torch_tensor'

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        tensor = super().validate(value=value, field=field, config=config)
        return VideoTensorMixin.validate_shape(cls, value=tensor)
