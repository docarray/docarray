from io import BytesIO
from typing import TYPE_CHECKING, Any, NamedTuple, Type, TypeVar

import numpy as np
from pydantic import parse_obj_as
from pydantic.validators import bytes_validator

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor import AudioNdArray, NdArray, VideoNdArray
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from pydantic.fields import BaseConfig, ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='VideoBytes')


class VideoLoadResult(NamedTuple):
    video: VideoNdArray
    audio: AudioNdArray
    key_frame_indices: NdArray


@_register_proto(proto_type_name='video_bytes')
class VideoBytes(bytes, AbstractType):
    """
    Bytes that store a video and that can be load into a video tensor
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        value = bytes_validator(value)
        return cls(value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    def load(self, **kwargs) -> VideoLoadResult:
        """
        Load the video from the bytes into a VideoLoadResult object consisting of a
        VideoNdArray (`VideoLoadResult.video`), an AudioNdArray
        (`VideoLoadResult.audio`) and an NdArray containing the key frame indices
        (`VideoLoadResult.key_frame_indices`).

        ---

        ```python
        from docarray import BaseDoc
        from docarray.typing import VideoUrl
        import numpy as np


        class MyDoc(BaseDoc):
            video_url: VideoUrl


        doc = MyDoc(
            video_url='https://github.com/docarray/docarray/blob/feat-rewrite-v2/tests/toydata/mov_bbb.mp4?raw=true'
        )

        video, audio, key_frame_indices = doc.video_url.load()
        assert isinstance(video, np.ndarray)
        assert isinstance(audio, np.ndarray)
        assert isinstance(key_frame_indices, np.ndarray)
        ```

        ---


        :param kwargs: supports all keyword arguments that are being supported by
            av.open() as described [here](https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open)
        :return: a VideoLoadResult instance with video, audio and keyframe indices
        """
        if TYPE_CHECKING:
            import av
        else:
            av = import_library('av')

        with av.open(BytesIO(self), **kwargs) as container:
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
