from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoNdArray')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


@_register_proto(proto_type_name='video_ndarray')
class VideoNdArray(NdArray, VideoTensorMixin):
    """
    Subclass of NdArray, to represent a video tensor.
    Adds video-specific features to the tensor.

    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        import numpy as np
        from pydantic import parse_obj_as

        from docarray import BaseDocument
        from docarray.typing import VideoNdArray, VideoUrl


        class MyVideoDoc(BaseDocument):
            title: str
            url: Optional[VideoUrl]
            video_tensor: Optional[VideoNdArray]


        doc_1 = MyVideoDoc(
            title='my_first_video_doc',
            video_tensor=np.random.random((100, 224, 224, 3)),
        )

        doc_1.video_tensor.save(file_path='file_1.mp4')


        doc_2 = MyVideoDoc(
            title='my_second_video_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc_2.video_tensor = parse_obj_as(VideoNdArray, doc_2.url.load().video)
        doc_2.video_tensor.save(file_path='file_2.mp4')

    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        tensor = super().validate(value=value, field=field, config=config)
        return cls.validate_shape(value=tensor)
