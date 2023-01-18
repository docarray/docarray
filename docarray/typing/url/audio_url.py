import wave
from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

import numpy as np
from pydantic import parse_obj_as

from docarray.typing.proto_register import register_proto
from docarray.typing.tensor.audio.audio_ndarray import MAX_INT_16, AudioNdArray
from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

T = TypeVar('T', bound='AudioUrl')

AUDIO_FILE_FORMATS = ['wav']


@register_proto(proto_type_name='audio_url')
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

    def load(self: T, dtype: str = 'float32') -> AudioNdArray:
        """
        Load the data from the url into an AudioNdArray.

        :param dtype: Data-type of the returned array; default: float32.
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
        import io

        file: Union[io.BytesIO, T]

        if self.startswith('http'):
            import requests

            resp = requests.get(self)
            resp.raise_for_status()
            file = io.BytesIO()
            file.write(resp.content)
            file.seek(0)
        else:
            file = self

        # note wave is Python built-in mod. https://docs.python.org/3/library/wave.html
        with wave.open(file) as ifile:
            samples = ifile.getnframes()
            audio = ifile.readframes(samples)

            # Convert buffer to float32 using NumPy
            audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
            audio_as_np_float32 = audio_as_np_int16.astype(dtype=dtype)

            # Normalise float32 array so that values are between -1.0 and +1.0
            audio_norm = audio_as_np_float32 / MAX_INT_16

            channels = ifile.getnchannels()
            if channels == 2:
                # 1 for mono, 2 for stereo
                audio_stereo = np.empty((int(len(audio_norm) / channels), channels))
                audio_stereo[:, 0] = audio_norm[range(0, len(audio_norm), 2)]
                audio_stereo[:, 1] = audio_norm[range(1, len(audio_norm), 2)]

                return parse_obj_as(AudioNdArray, audio_stereo)
            else:
                return parse_obj_as(AudioNdArray, audio_norm)
