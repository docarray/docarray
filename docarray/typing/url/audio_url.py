import wave
from typing import TYPE_CHECKING, Any, Type, TypeVar, Union

import numpy as np

from docarray.typing.url.any_url import AnyUrl

if TYPE_CHECKING:
    from pydantic import BaseConfig
    from pydantic.fields import ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='AudioUrl')

AUDIO_FILE_FORMATS = ['wav']
MAX_INT_16 = 2**15


class AudioUrl(AnyUrl):
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

        return NodeProto(audio_url=str(self))

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[T, np.ndarray, Any],
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:
        url = super().validate(value, field, config)  # basic url validation
        has_audio_extension = any(url.endswith(ext) for ext in AUDIO_FILE_FORMATS)
        if not has_audio_extension:
            raise ValueError(
                f'Audio URL must have one of the following extensions:'
                f'{AUDIO_FILE_FORMATS}'
            )
        return cls(str(url), scheme=None)

    def load(self: T) -> np.ndarray:
        """
        Load the data from the url into a numpy.ndarray.

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import Document
            import numpy as np

            from docarray.typing import AudioUrl


            class MyDoc(Document):
                audio_url: AudioUrl


            doc = MyDoc(audio_url="toydata/hello.wav")

            audio_tensor = doc.audio_url.load()
            assert isinstance(audio_tensor, np.ndarray)

        :return: np.ndarray representing the audio file content
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
            audio_as_np_float32 = audio_as_np_int16.astype(np.float32)

            # Normalise float32 array so that values are between -1.0 and +1.0
            audio_norm = audio_as_np_float32 / MAX_INT_16

            channels = ifile.getnchannels()
            if channels == 2:
                # 1 for mono, 2 for stereo
                audio_stereo = np.empty((int(len(audio_norm) / channels), channels))
                audio_stereo[:, 0] = audio_norm[range(0, len(audio_norm), 2)]
                audio_stereo[:, 1] = audio_norm[range(1, len(audio_norm), 2)]

                return audio_stereo
            else:
                return audio_norm
