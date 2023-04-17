from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_doc import BaseDoc
from docarray.documents import AudioDoc
from docarray.typing import AnyEmbedding, AnyTensor, VideoBytes
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.typing.url.video_url import VideoUrl
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
else:
    tf = import_library('tensorflow', raise_error=False)
    torch = import_library('torch', raise_error=False)


T = TypeVar('T', bound='VideoDoc')


class VideoDoc(BaseDoc):
    """
    Document for handling video.

    The Video Document can contain:

    - a [`VideoUrl`][docarray.typing.url.VideoUrl] (`VideoDoc.url`)
    - an [`AudioDoc`][docarray.documents.AudioDoc] (`VideoDoc.audio`)
    - a [`VideoTensor`](../../../api_references/typing/tensor/video) (`VideoDoc.tensor`)
    - an [`AnyTensor`](../../../api_references/typing/tensor/tensor) representing the indices of the video's key frames (`VideoDoc.key_frame_indices`)
    - an [`AnyEmbedding`](../../../api_references/typing/tensor/embedding) (`VideoDoc.embedding`)
    - a [`VideoBytes`][docarray.typing.bytes.VideoBytes] object (`VideoDoc.bytes_`)

    You can use this Document directly:

    ```python
    from docarray.documents import VideoDoc

    # use it directly
    vid = VideoDoc(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
    )
    vid.tensor, vid.audio.tensor, vid.key_frame_indices = vid.url.load()
    # model = MyEmbeddingModel()
    # vid.embedding = model(vid.tensor)
    ```

    You can extend this Document:

    ```python
    from typing import Optional

    from docarray.documents import TextDoc, VideoDoc


    # extend it
    class MyVideo(VideoDoc):
        name: Optional[TextDoc]


    video = MyVideo(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
    )
    video.name = TextDoc(text='my first video')
    video.tensor = video.url.load().video
    # model = MyEmbeddingModel()
    # video.embedding = model(video.tensor)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import TextDoc, VideoDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        video: VideoDoc
        text: TextDoc


    mmdoc = MultiModalDoc(
        video=VideoDoc(
            url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
        ),
        text=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.video.tensor = mmdoc.video.url.load().video

    # or
    mmdoc.video.bytes_ = mmdoc.video.url.load_bytes()
    mmdoc.video.tensor = mmdoc.video.bytes_.load().video
    ```
    """

    url: Optional[VideoUrl]
    audio: Optional[AudioDoc] = AudioDoc()
    tensor: Optional[VideoTensor]
    key_frame_indices: Optional[AnyTensor]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[VideoBytes]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(url=value)
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch is not None
            and isinstance(value, torch.Tensor)
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = cls(tensor=value)

        return super().validate(value)
