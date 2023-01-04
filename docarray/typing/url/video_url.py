from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar, Union

import numpy as np
from pydantic.tools import parse_obj_as

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
        self: T, only_keyframes: bool = False, **kwargs
    ) -> Union[VideoNdArray, Tuple[VideoNdArray, np.ndarray]]:
        """
        Load the data from the url into a VideoNdArray or Tuple of VideoNdArray and
        np.ndarray.



        :param only_keyframes: if True keep only the keyframes, if False keep all frames
            and store the indices of the keyframes in :attr:`.tags`
        :param kwargs: supports all keyword arguments that are being supported by
            av.open() as described in:
            https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open
        :return: np.ndarray representing the audio file content, list of key frame
            indices if only_keyframe False.
        """
        import av

        with av.open(self, **kwargs) as container:
            if only_keyframes:
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = 'NONKEY'

            frames = []
            keyframe_indices = []
            for i, frame in enumerate(container.decode(video=0)):

                img = frame.to_image()
                frames.append(img)
                if not only_keyframes and frame.key_frame == 1:
                    keyframe_indices.append(i)

        frames_vid = parse_obj_as(VideoNdArray, np.moveaxis(np.stack(frames), -3, -2))

        if only_keyframes:
            return frames_vid
        else:
            return frames_vid, np.ndarray(keyframe_indices)
