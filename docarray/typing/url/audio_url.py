import warnings
from typing import List, Optional, Tuple, TypeVar

from docarray.typing import AudioNdArray
from docarray.typing.bytes.audio_bytes import AudioBytes
from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.mimetypes import AUDIO_MIMETYPE
from docarray.utils._internal.misc import is_notebook

T = TypeVar('T', bound='AudioUrl')


@_register_proto(proto_type_name='audio_url')
class AudioUrl(AnyUrl):
    """
    URL to an audio file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def mime_type(cls) -> str:
        return AUDIO_MIMETYPE

    @classmethod
    def extra_extensions(cls) -> List[str]:
        """
        Returns a list of additional file extensions that are valid for this class
        but cannot be identified by the mimetypes library.
        """
        return []

    def load(self: T) -> Tuple[AudioNdArray, int]:
        """
        Load the data from the url into an [`AudioNdArray`][docarray.typing.AudioNdArray]
        and the frame rate.

        ---

        ```python
        from typing import Optional

        from docarray import BaseDoc
        from docarray.typing import AudioNdArray, AudioUrl


        class MyDoc(BaseDoc):
            audio_url: AudioUrl
            audio_tensor: Optional[AudioNdArray] = None


        doc = MyDoc(audio_url='https://www.kozco.com/tech/piano2.wav')
        doc.audio_tensor, _ = doc.audio_url.load()
        assert isinstance(doc.audio_tensor, AudioNdArray)
        ```

        ---

        :return: tuple of an [`AudioNdArray`][docarray.typing.AudioNdArray] representing
            the audio file content, and an integer representing the frame rate.

        """
        bytes_ = self.load_bytes()
        return bytes_.load()

    def load_bytes(self, timeout: Optional[float] = None) -> AudioBytes:
        """
        Convert url to [`AudioBytes`][docarray.typing.AudioBytes]. This will either load or
        download the file and save it into an [`AudioBytes`][docarray.typing.AudioBytes] object.

        :param timeout: timeout for urlopen. Only relevant if url is not local
        :return: [`AudioBytes`][docarray.typing.AudioBytes] object
        """
        bytes_ = super().load_bytes(timeout=timeout)
        return AudioBytes(bytes_)

    def display(self):
        """
        Play the audio sound from url in notebook.
        """
        if is_notebook():
            from IPython.display import Audio, display

            remote_url = True if self.startswith('http') else False

            if remote_url:
                display(Audio(data=self))
            else:
                display(Audio(filename=self))
        else:
            warnings.warn('Display of audio is only possible in a notebook.')
