from typing import TYPE_CHECKING, Any, NamedTuple, Type, TypeVar, Union

import numpy as np
from pydantic.tools import parse_obj_as

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio.audio_ndarray import AudioNdArray
from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.video import VideoNdArray
from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='VideoUrl')

VIDEO_FILE_FORMATS = ['mp4']


class VideoLoadResult(NamedTuple):
    video: VideoNdArray
    audio: AudioNdArray
    key_frame_indices: NdArray


@_register_proto(proto_type_name='video_url')
class VideoUrl(AnyUrl):
    """
    URL to a .wav file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        url = super().validate(value, field, config)
        has_video_extension = any(ext in url for ext in VIDEO_FILE_FORMATS)
        if not has_video_extension:
            raise ValueError(
                f'Video URL must have one of the following extensions:'
                f'{VIDEO_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

    def load(self: T, **kwargs) -> VideoLoadResult:
        """
        Load the data from the url into a named Tuple of VideoNdArray, AudioNdArray and
        NdArray.

        :param kwargs: supports all keyword arguments that are being supported by
            av.open() as described in:
            https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open

        :return: AudioNdArray representing the audio content, VideoNdArray representing
            the images of the video, NdArray of the key frame indices.


        EXAMPLE USAGE

        .. code-block:: python

            from typing import Optional

            from docarray import BaseDocument

            from docarray.typing import VideoUrl, VideoNdArray, AudioNdArray, NdArray


            class MyDoc(BaseDocument):
                video_url: VideoUrl
                video: Optional[VideoNdArray]
                audio: Optional[AudioNdArray]
                key_frame_indices: Optional[NdArray]


            doc = MyDoc(
                video_url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
            )
            doc.video, doc.audio, doc.key_frame_indices = doc.video_url.load()

            assert isinstance(doc.video, VideoNdArray)
            assert isinstance(doc.audio, AudioNdArray)
            assert isinstance(doc.key_frame_indices, NdArray)

        You can load only the key frames (or video, audio respectively):

        .. code-block:: python

            from pydantic import parse_obj_as

            from docarray.typing import NdArray, VideoUrl


            url = parse_obj_as(
                VideoUrl,
                'https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true',
            )
            key_frame_indices = url.load().key_frame_indices
            assert isinstance(key_frame_indices, NdArray)

        """
        import av

        with av.open(self, **kwargs) as container:
            audio_frames = []
            video_frames = []
            keyframe_indices = []

            for frame in container.decode():
                if type(frame) == av.audio.frame.AudioFrame:
                    audio_frames.append(frame.to_ndarray())
                elif type(frame) == av.video.frame.VideoFrame:
                    video_frames.append(frame.to_ndarray(format='rgb24'))

                    if frame.key_frame == 1:
                        curr_index = len(video_frames)
                        keyframe_indices.append(curr_index)

        if len(audio_frames) == 0:
            audio = parse_obj_as(AudioNdArray, np.array(audio_frames))
        else:
            audio = parse_obj_as(AudioNdArray, np.stack(audio_frames))

        video = parse_obj_as(VideoNdArray, np.stack(video_frames))
        indices = parse_obj_as(NdArray, keyframe_indices)

        return VideoLoadResult(video=video, audio=audio, key_frame_indices=indices)
