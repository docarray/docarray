from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.jaxarray import JaxArray, metaJax
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoJaxArray')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


@_register_proto(proto_type_name='video_jaxarray')
class VideoJaxArray(JaxArray, VideoTensorMixin, metaclass=metaJax):
    """ """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        tensor = super().validate(value=value, field=field, config=config)
        return cls.validate_shape(value=tensor)
