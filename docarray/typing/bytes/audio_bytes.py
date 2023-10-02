import io
from typing import Tuple, TypeVar

import numpy as np
from pydantic import parse_obj_as

from docarray.typing.bytes.base_bytes import BaseBytes
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio import AudioNdArray
from docarray.utils._internal.misc import import_library

T = TypeVar('T', bound='AudioBytes')


@_register_proto(proto_type_name='audio_bytes')
class AudioBytes(BaseBytes):
    """
    Bytes that store an audio and that can be load into an Audio tensor
    """

    def load(self) -> Tuple[AudioNdArray, int]:
        """
        Load the Audio from the [`AudioBytes`][docarray.typing.AudioBytes] into an
        [`AudioNdArray`][docarray.typing.AudioNdArray].

        ---

        ```python
        from typing import Optional
        from docarray import BaseDoc
        from docarray.typing import AudioBytes, AudioNdArray, AudioUrl


        class MyAudio(BaseDoc):
            url: AudioUrl
            tensor: Optional[AudioNdArray] = None
            bytes_: Optional[AudioBytes] = None
            frame_rate: Optional[float] = None


        doc = MyAudio(url='https://www.kozco.com/tech/piano2.wav')
        doc.bytes_ = doc.url.load_bytes()
        doc.tensor, doc.frame_rate = doc.bytes_.load()

        # Note this is equivalent to do

        doc.tensor, doc.frame_rate = doc.url.load()

        assert isinstance(doc.tensor, AudioNdArray)
        ```

        ---
        :return: tuple of an [`AudioNdArray`][docarray.typing.AudioNdArray] representing the
            audio bytes content, and an integer representing the frame rate.
        """
        pydub = import_library('pydub', raise_error=True)  # noqa: F841
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(self))

        # Convert to float32 using NumPy
        samples = np.array(segment.get_array_of_samples())

        # Normalise float32 array so that values are between -1.0 and +1.0
        samples_norm = samples / 2 ** (segment.sample_width * 8 - 1)
        return parse_obj_as(AudioNdArray, samples_norm), segment.frame_rate
