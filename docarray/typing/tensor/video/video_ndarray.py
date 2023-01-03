from typing import TypeVar

import numpy as np

from docarray.typing.tensor.ndarray import NdArray
from docarray.typing.tensor.video.abstract_video_tensor import AbstractVideoTensor

T = TypeVar('T', bound='VideoNdArray')


class VideoNdArray(AbstractVideoTensor, NdArray):
    """
    Subclass of NdArray, to represent a video tensor.

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
            url='https://github.com/docarray/docarray/tree/feat-add-audio-v2/tests/toydata/hello.wav',
        )
        doc_2.audio_tensor = parse_obj_as(AudioNdArray, doc_2.url.load())
        doc_2.audio_tensor.save_to_wav_file(file_path='path/to/file_2.wav')
    """

    _PROTO_FIELD_NAME = 'video_ndarray'

    def check_shape(self) -> None:
        if self.ndim != 4 or self.shape[-1] != 3 or self.dtype != np.uint8:
            raise ValueError(
                f'expects `` with dtype=uint8 and ndim=4 and the last dimension is 3, '
                f'but receiving {self.shape} in {self.dtype}'
            )

    def to_numpy(self) -> np.ndarray:
        return self
