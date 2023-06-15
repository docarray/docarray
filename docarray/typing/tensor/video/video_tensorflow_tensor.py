from typing import Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor, metaTensorFlow
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoTensorFlowTensor')


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
    # doc_1.video_tensor.save(file_path='file_1.mp4')

    doc_2 = MyVideoDoc(
        title='my_second_video_doc',
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true',
    )

    doc_2.video_tensor = doc_2.url.load().video
    # doc_2.video_tensor.save(file_path='file_2.wav')
    ```

    ---
    """

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Union[T, np.ndarray, List[Any], Tuple[Any], Any],
    ) -> T:
        tensor = super()._docarray_validate(value=value)
        return cls.validate_shape(value=tensor)
