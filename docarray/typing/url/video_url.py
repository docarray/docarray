import warnings
from typing import List, Optional, TypeVar

from docarray.typing.bytes.video_bytes import VideoBytes, VideoLoadResult
from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.mimetypes import VIDEO_MIMETYPE
from docarray.utils._internal.misc import is_notebook

T = TypeVar('T', bound='VideoUrl')


@_register_proto(proto_type_name='video_url')
class VideoUrl(AnyUrl):
    """
    URL to a video file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def mime_type(cls) -> str:
        return VIDEO_MIMETYPE

    @classmethod
    def extra_extensions(cls) -> List[str]:
        """
        Returns a list of additional file extensions that are valid for this class
        but cannot be identified by the mimetypes library.
        """
        return []

    def load(self: T, **kwargs) -> VideoLoadResult:
        """
        Load the data from the url into a `NamedTuple` of
        [`VideoNdArray`][docarray.typing.VideoNdArray],
        [`AudioNdArray`][docarray.typing.AudioNdArray]
        and [`NdArray`][docarray.typing.NdArray].

        ---

        ```python
        from typing import Optional

        from docarray import BaseDoc

        from docarray.typing import VideoUrl, VideoNdArray, AudioNdArray, NdArray


        class MyDoc(BaseDoc):
            video_url: VideoUrl
            video: Optional[VideoNdArray] = None
            audio: Optional[AudioNdArray] = None
            key_frame_indices: Optional[NdArray] = None


        doc = MyDoc(
            video_url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
        )
        doc.video, doc.audio, doc.key_frame_indices = doc.video_url.load()

        assert isinstance(doc.video, VideoNdArray)
        assert isinstance(doc.audio, AudioNdArray)
        assert isinstance(doc.key_frame_indices, NdArray)
        ```

        ---

        You can load only the key frames (or video, audio respectively):

        ---

        ```python
        from pydantic import parse_obj_as

        from docarray.typing import NdArray, VideoUrl


        url = parse_obj_as(
            VideoUrl,
            'https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true',
        )
        key_frame_indices = url.load().key_frame_indices
        assert isinstance(key_frame_indices, NdArray)
        ```

        ---

        :param kwargs: supports all keyword arguments that are being supported by
            av.open() as described [here](https://pyav.org/docs/stable/api/_globals.html?highlight=open#av.open)

        :return: [`AudioNdArray`][docarray.typing.AudioNdArray] representing the audio content,
            [`VideoNdArray`][docarray.typing.VideoNdArray] representing the images of the video,
            [`NdArray`][docarray.typing.NdArray] of the key frame indices.
        """
        buffer = self.load_bytes(**kwargs)
        return buffer.load()

    def load_bytes(self, timeout: Optional[float] = None) -> VideoBytes:
        """
        Convert url to [`VideoBytes`][docarray.typing.VideoBytes]. This will either load or download
        the file and save it into an [`VideoBytes`][docarray.typing.VideoBytes] object.

        :param timeout: timeout for urlopen. Only relevant if url is not local
        :return: [`VideoBytes`][docarray.typing.VideoBytes] object
        """
        bytes_ = super().load_bytes(timeout=timeout)
        return VideoBytes(bytes_)

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
