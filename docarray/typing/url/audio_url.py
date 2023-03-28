import warnings
from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar, Union

import numpy as np

from docarray.typing.bytes.audio_bytes import AudioBytes
from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.typing.url.filetypes import AUDIO_FILE_FORMATS
from docarray.utils.misc import is_notebook

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='AudioUrl')


@_register_proto(proto_type_name='audio_url')
class AudioUrl(AnyUrl):
    """
    URL to a audio file.
    Can be remote (web) URL, or a local file path.
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, str, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        import os
        from urllib.parse import urlparse

        url = super().validate(value, field, config)  # basic url validation
        path = urlparse(url).path
        ext = os.path.splitext(path)[1][1:].lower()

        # pass test if extension is valid or no extension
        has_audio_extension = ext in AUDIO_FILE_FORMATS or ext == ''

        if not has_audio_extension:
            raise ValueError('Audio URL must have a valid extension')
        return cls(str(url), scheme=None)

    def load(self: T) -> Tuple[np.ndarray, int]:
        """
        Load the data from the url into an AudioNdArray.

        :return: AudioNdArray representing the audio file content.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDoc
            import numpy as np

            from docarray.typing import AudioUrl


            class MyDoc(Document):
                audio_url: AudioUrl
                audio_tensor: AudioNdArray


            doc = MyDoc(audio_url="toydata/hello.wav")
            doc.audio_tensor, doc.frame_rate = doc.audio_url.load()
            assert isinstance(doc.audio_tensor, np.ndarray)

        """
        bytes_ = AudioBytes(self.load_bytes())
        return bytes_.load()

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
            warnings.warn('Display of image is only possible in a notebook.')
