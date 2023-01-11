from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic.tools import parse_obj_as

from docarray.typing import AudioNdArray, NdArray
from docarray.typing.tensor.video import VideoNdArray
from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='VideoUrl')

VIDEO_FILE_FORMATS = ['mp4']


class VideoUrl(AnyUrl):
    """
    URL to a .wav file.
    Can be remote (web) URL, or a local file path.
    """

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """Convert Document into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that needs to
        be converted into a protobuf
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        return NodeProto(video_url=str(self))

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

    def load(
        self: T, only_keyframes: bool = False, audio_format: str = 'fltp', **kwargs
    ) -> Union[VideoNdArray, Tuple[AudioNdArray, VideoNdArray, NdArray]]:
        """
        Load the data from the url into a VideoNdArray or Tuple of AudioNdArray,
        VideoNdArray and NdArray.

        :param only_keyframes: if True keep only the keyframes, if False keep all frames
            and store the indices of the keyframes in :attr:`.tags`
        :param kwargs: supports all keyword arguments that are being supported by
            av.open() as described in:
            https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open
        :return: AudioNdArray representing the audio content, VideoNdArray representing
            the images of the video, NdArray of key frame indices if only_keyframe
            False, else only VideoNdArray representing the keyframes.
        """
        import av

        with av.open(self, **kwargs) as container:
            if only_keyframes:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'

            audio_frames = []
            video_frames = []
            keyframe_indices = []

            for frame in container.decode():
                if type(frame) == av.audio.frame.AudioFrame:
                    audio_frames.append(frame.to_ndarray(format=audio_format))
                elif type(frame) == av.video.frame.VideoFrame:
                    video_frames.append(frame.to_ndarray(format='rgb24'))

                    if not only_keyframes and frame.key_frame == 1:
                        curr_index = len(video_frames)
                        keyframe_indices.append(curr_index)

        video = parse_obj_as(VideoNdArray, np.stack(video_frames))

        if only_keyframes:
            return video
        else:
            audio = parse_obj_as(AudioNdArray, np.stack(audio_frames))
            indices = parse_obj_as(NdArray, keyframe_indices)
            return audio, video, indices
