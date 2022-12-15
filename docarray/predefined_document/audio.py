import wave
from typing import BinaryIO, Optional, TypeVar, Union

from docarray.document import BaseDocument
from docarray.typing import AudioUrl, Embedding, Tensor

T = TypeVar('T', bound='Audio')


class Audio(BaseDocument):
    """
    Document for handling audios.

    The Audio Document can contain an AudioUrl (`Audio.url`), a Tensor
    (`Audio.tensor`), and an Embedding (`Audio.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import Audio

        # use it directly
        audio = Audio(url='https://www.kozco.com/tech/piano2.wav')
        audio.tensor = audio.url.load()
        model = MyEmbeddingModel()
        audio.embedding = model(audio.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray import Audio
        from docarray.typing import Embedding
        from typing import Optional

        # extend it
        class MyAudio(Audio):
            name: Optional[Text]


        audio = MyAudio(url='https://www.kozco.com/tech/piano2.wav')
        audio.tensor = audio.url.load()
        model = MyEmbeddingModel()
        audio.embedding = model(audio.tensor)
        audio.name = 'my first audio'


    You can use this Document for composition:

    .. code-block:: python

        from docarray import Document, Audio, Text

        # compose it
        class MultiModalDoc(Document):
            audio: Audio
            text: Text


        mmdoc = MultiModalDoc(
            audio=Audio(url='https://www.kozco.com/tech/piano2.wav'),
            text=Text(text='hello world, how are you doing?'),
        )
        mmdoc.audio.tensor = mmdoc.audio.url.load()
    """

    url: Optional[AudioUrl]
    tensor: Optional[Tensor]
    embedding: Optional[Embedding]

    def save_audio_tensor_to_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        sample_rate: int = 44100,
        sample_width: int = 2,
    ) -> None:
        """Save :attr:`.tensor` into a .wav file. Mono/stereo is preserved.

        :param file_path: if file is a string, open the file by that name, otherwise
            treat it as a file-like object.
        :param sample_rate: sampling frequency
        :param sample_width: sample width in bytes
        """
        if self.tensor is None:
            raise ValueError(
                'Audio.tensor has not been set, and therefore cannot be saved to file.'
            )

        # Convert to (little-endian) 16 bit integers.
        max_int16 = 2**15
        tensor = (self.tensor * max_int16).astype('<h')
        n_channels = 2 if self.tensor.ndim > 1 else 1

        with wave.open(file_path, 'w') as f:
            # 2 Channels.
            f.setnchannels(n_channels)
            # 2 bytes per sample.
            f.setsampwidth(sample_width)
            f.setframerate(sample_rate)
            f.writeframes(tensor.tobytes())
