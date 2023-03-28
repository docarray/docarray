import warnings
from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

from docarray.typing.bytes.video_bytes import VideoLoadResult
from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.utils._internal.misc import is_notebook

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='VideoUrl')

VIDEO_FILE_FORMATS = ['mp4']


@_register_proto(proto_type_name='video_url')
class VideoUrl(AnyUrl):
    """
    URL to a .wav file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, str, Any],
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

            from docarray import BaseDoc

            from docarray.typing import VideoUrl, VideoNdArray, AudioNdArray, NdArray


            class MyDoc(BaseDoc):
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
        from docarray.typing.bytes.video_bytes import VideoBytes

        buffer = VideoBytes(self.load_bytes(**kwargs))
        return buffer.load()

    def display(self):
        """
        Play video from url in notebook.
        """
        if is_notebook():
            from IPython.display import display

            remote_url = True if self.startswith('http') else False

            if remote_url:
                from IPython.display import Video

                b = self.load_bytes()
                display(Video(data=b, embed=True, mimetype='video/mp4'))
            else:
                import os

                from IPython.display import HTML

                path = os.path.relpath(self)
                src = f'''
                    <body>
                    <video width="320" height="240" autoplay muted controls>
                    <source src="{path}">
                    Your browser does not support the video tag.
                    </video>
                    </body>
                    '''
                display(HTML(src))

        else:
            warnings.warn('Display of video is only possible in a notebook.')
