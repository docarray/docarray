from typing import Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_document import BaseDocument
from docarray.documents import AudioDoc
from docarray.typing import AnyEmbedding, AnyTensor
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.typing.url.video_url import VideoUrl
from docarray.utils.misc import is_tf_available, is_torch_available

torch_available = is_torch_available()
if torch_available:
    import torch


tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore


T = TypeVar('T', bound='VideoDoc')


class VideoDoc(BaseDocument):
    """
    Document for handling video.
    The Video Document can contain a VideoUrl (`VideoDoc.url`), an Audio Document
    (`VideoDoc.audio`), a VideoTensor (`VideoDoc.tensor`), an AnyTensor representing
    the indices of the video's key frames (`VideoDoc.key_frame_indices`) and an
    AnyEmbedding (`VideoDoc.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import Video

        # use it directly
        vid = Video(
            url='https://github.com/docarray/docarray/tree/feat-add-video-v2/tests/toydata/mov_bbb.mp4?raw=true'
        )
        vid.audio.tensor, vid.tensor, vid.key_frame_indices = vid.url.load()
        model = MyEmbeddingModel()
        vid.embedding = model(vid.tensor)

    You can extend this Document:

    .. code-block:: python

        from typing import Optional

        from docarray.documents import TextDoc, VideoDoc


        # extend it
        class MyVideo(Video):
            name: Optional[Text]


        video = MyVideo(
            url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
        )
        video.video_tensor = video.url.load().video
        model = MyEmbeddingModel()
        video.embedding = model(video.tensor)
        video.name = Text(text='my first video')

    You can use this Document for composition:

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.documents import TextDoc, VideoDoc


        # compose it
        class MultiModalDoc(BaseDocument):
            video: Video
            text: Text


        mmdoc = MultiModalDoc(
            video=Video(
                url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
            ),
            text=Text(text='hello world, how are you doing?'),
        )
        mmdoc.video.video_tensor = mmdoc.video.url.load().video

        # or

        mmdoc.video.bytes_ = mmdoc.video.url.load_bytes()

    """

    url: Optional[VideoUrl]
    audio: Optional[AudioDoc] = AudioDoc()
    tensor: Optional[VideoTensor]
    key_frame_indices: Optional[AnyTensor]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[bytes]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(url=value)
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch_available
            and isinstance(value, torch.Tensor)
            or (tf_available and isinstance(value, tf.Tensor))
        ):
            value = cls(tensor=value)

        return super().validate(value)
