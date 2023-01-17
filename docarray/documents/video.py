from typing import Optional, TypeVar

from docarray.base_document import BaseDocument
from docarray.documents import Audio
from docarray.typing import AnyEmbedding, AnyTensor
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.typing.url.video_url import VideoUrl

T = TypeVar('T', bound='Video')


class Video(BaseDocument):
    """
    Document for handling video.
    The Video Document can contain a VideoUrl (`Video.url`), an Audio Document
    (`Video.audio`), a VideoTensor (`Video.video_tensor`), an AnyTensor representing
    the indices of the video's key frames (`Video.key_frame_indices`) and an
    AnyEmbedding (`Video.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import Video

        # use it directly
        vid = Video(
            url='https://github.com/docarray/docarray/tree/feat-add-video-v2/tests/toydata/mov_bbb.mp4?raw=true'
        )
        vid.audio.tensor, vid.video_tensor, vid.key_frame_indices = vid.url.load()
        model = MyEmbeddingModel()
        vid.embedding = model(vid.video_tensor)

    You can extend this Document:

    .. code-block:: python

        from typing import Optional

        from docarray.documents import Text, Video


        # extend it
        class MyVideo(Video):
            name: Optional[Text]


        video = MyVideo(
            url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
        )
        video.video_tensor = video.url.load_key_frames()
        model = MyEmbeddingModel()
        video.embedding = model(video.video_tensor)
        video.name = Text(text='my first video')

    You can use this Document for composition:

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.documents import Text, Video


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
        mmdoc.video.video_tensor = mmdoc.video.url.load_key_frames()
    """

    url: Optional[VideoUrl]
    audio: Optional[Audio] = Audio()
    video_tensor: Optional[VideoTensor]
    key_frame_indices: Optional[AnyTensor]
    embedding: Optional[AnyEmbedding]
