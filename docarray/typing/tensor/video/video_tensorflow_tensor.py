from typing import TYPE_CHECKING, Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor, metaTensorFlow
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoTensorFlowTensor')

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField


@_register_proto(proto_type_name='video_tensorflow_tensor')
class VideoTensorFlowTensor(
    TensorFlowTensor, VideoTensorMixin, metaclass=metaTensorFlow
):
    """
    Subclass of [`TensorFlowTensor`][docarray.typing.TensorFlowTensor],
    to represent a video tensor. Adds video-specific features to the tensor.

    ---

    ```python
    from typing import Optional

    import tensorflow as tf
    from pydantic import parse_obj_as

    from docarray import BaseDoc
    from docarray.typing import VideoTensorFlowTensor, VideoUrl


    class MyVideoDoc(BaseDoc):
        title: str
        url: Optional[VideoUrl]
        video_tensor: Optional[VideoTensorFlowTensor]


    doc_1 = MyVideoDoc(
        title='my_first_video_doc',
        video_tensor=tf.random.normal((100, 224, 224, 3)),
    )

    doc_1.video_tensor.save(file_path='file_1.wav')


    doc_2 = MyVideoTESTDoc(
        title='my_second_video_doc',
        url='https://www.kozco.com/tech/piano2.wav',
    )

    doc_2.video_tensor = parse_obj_as(VideoTensorFlowTensor, doc_2.url.load())
    doc_2.video_tensor.save(file_path='file_2.wav')
    ```

    ---
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
