from typing import Optional, TypeVar

from docarray.document_base import BaseDocument
from docarray.typing import AnyEmbedding, AudioUrl
from docarray.typing.tensor.audio.audio_tensor import AudioTensor

T = TypeVar('T', bound='Audio')


class Audio(BaseDocument):
    """
    Document for handling audios.

    The Audio Document can contain an AudioUrl (`Audio.url`), an AudioTensor
    (`Audio.tensor`), and an AnyEmbedding (`Audio.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import Audio

        # use it directly
        audio = Audio(
            url='https://github.com/docarray/docarray/tree/feat-add-audio-v2/tests/toydata/hello.wav?raw=true'
        )
        audio.tensor = audio.url.load()
        model = MyEmbeddingModel()
        audio.embedding = model(audio.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray import Audio, Text
        from typing import Optional

        # extend it
        class MyAudio(Audio):
            name: Optional[Text]


        audio = MyAudio(
            url='https://github.com/docarray/docarray/tree/feat-add-audio-v2/tests/toydata/hello.wav?raw=true'
        )
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
            audio=Audio(
                url='https://github.com/docarray/docarray/tree/feat-add-audio-v2/tests/toydata/hello.wav?raw=true'
            ),
            text=Text(text='hello world, how are you doing?'),
        )
        mmdoc.audio.tensor = mmdoc.audio.url.load()
    """

    url: Optional[AudioUrl]
    tensor: Optional[AudioTensor]
    embedding: Optional[AnyEmbedding]
