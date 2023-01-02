from typing import TypeVar

from docarray.typing import NdArray
from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor

MAX_INT_16 = 2**15

T = TypeVar('T', bound='AudioNdArray')


class AudioNdArray(AbstractAudioTensor, NdArray):
    """
    Subclass of NdArray, to represent an audio tensor.
    Additionally, this allows storing such a tensor as a .wav audio file.


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        from pydantic import parse_obj_as

        from docarray import Document
        from docarray.typing import AudioNdArray, AudioUrl
        import numpy as np


        class MyAudioDoc(Document):
            title: str
            audio_tensor: Optional[AudioNdArray]
            url: Optional[AudioUrl]


        # from tensor
        doc_1 = MyAudioDoc(
            title='my_first_audio_doc',
            audio_tensor=np.random.rand(1000, 2),
        )

        doc_1.audio_tensor.save_to_wav_file(file_path='path/to/file_1.wav')

        # from url
        doc_2 = MyAudioDoc(
            title='my_second_audio_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc_2.audio_tensor = parse_obj_as(AudioNdArray, doc_2.url.load())
        doc_2.audio_tensor.save_to_wav_file(file_path='path/to/file_2.wav')

    """

    _PROTO_FIELD_NAME = 'audio_ndarray'

    def to_audio_bytes(self):
        tensor = (self * MAX_INT_16).astype('<h')
        return tensor.tobytes()
