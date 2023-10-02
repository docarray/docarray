from typing import Any, List, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.video.video_tensor_mixin import VideoTensorMixin

T = TypeVar('T', bound='VideoNdArray')


@_register_proto(proto_type_name='video_ndarray')
class VideoNdArray(NdArray, VideoTensorMixin):
    """
    Subclass of [`NdArray`][docarray.typing.NdArray], to represent a video tensor.
    Adds video-specific features to the tensor.

    ---

    ```python
    from typing import Optional

    import numpy as np
    from pydantic import parse_obj_as

    from docarray import BaseDoc
    from docarray.typing import VideoNdArray, VideoUrl


    class MyVideoDoc(BaseDoc):
        title: str
        url: Optional[VideoUrl] = None
        video_tensor: Optional[VideoNdArray] = None


    doc_1 = MyVideoDoc(
        title='my_first_video_doc',
        video_tensor=np.random.random((100, 224, 224, 3)),
    )

    doc_2 = MyVideoDoc(
        title='my_second_video_doc',
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true',
    )

    doc_2.video_tensor = parse_obj_as(VideoNdArray, doc_2.url.load().video)
    # doc_2.video_tensor.save(file_path='/tmp/file_2.mp4')
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
