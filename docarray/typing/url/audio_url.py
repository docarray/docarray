import warnings
from io import BytesIO
from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

import numpy as np

from docarray.typing.proto_register import _register_proto
from docarray.typing.url.any_url import AnyUrl
from docarray.utils.misc import is_notebook

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='AudioUrl')

AUDIO_FILE_FORMATS = ['wav']


@_register_proto(proto_type_name='audio_url')
class AudioUrl(AnyUrl):
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
        url = super().validate(value, field, config)  # basic url validation
        has_audio_extension = any(ext in url for ext in AUDIO_FILE_FORMATS)
        if not has_audio_extension:
            raise ValueError(
                f'Audio URL must have one of the following extensions:'
                f'{AUDIO_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

    def load(self: T) -> np.ndarray:
        """
        Load the data from the url into an AudioNdArray.

        :return: AudioNdArray representing the audio file content.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            import numpy as np

            from docarray.typing import AudioUrl


            class MyDoc(Document):
                audio_url: AudioUrl
                audio_tensor: AudioNdArray


            doc = MyDoc(audio_url="toydata/hello.wav")
            doc.audio_tensor = doc.audio_url.load()
            assert isinstance(doc.audio_tensor, np.ndarray)

        """
        from pydub import AudioSegment  # type: ignore

        bytes_ = self.load_bytes()
        segment = AudioSegment.from_file(BytesIO(bytes_))

        samples = segment.get_array_of_samples()
        samples = np.array(samples)

        samples_norm = samples / 2 ** (segment.sample_width * 8 - 1)
        return samples_norm

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
