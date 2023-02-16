import io
import wave
from typing import TYPE_CHECKING, Any, Type, TypeVar

import numpy as np
from pydantic import parse_obj_as
from pydantic.validators import bytes_validator

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio.abstract_audio_tensor import MAX_INT_16

if TYPE_CHECKING:
    from pydantic.fields import BaseConfig, ModelField

    from docarray.proto import NodeProto

T = TypeVar('T', bound='AudioBytes')


@_register_proto(proto_type_name='audio_bytes')
class AudioBytes(bytes, AbstractType):
    """
    Bytes that store an audio and that can be load into an Audio tensor
    """

    @classmethod
    def validate(
        cls: Type[T],
        value: Any,
        field: 'ModelField',
        config: 'BaseConfig',
    ) -> T:

        value = bytes_validator(value)
        return cls(value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    def load(self) -> np.ndarray:
        """
        Load the Audio from the bytes into a numpy.ndarray Audio tensor

        EXAMPLE USAGE

        .. code-block:: python

            from docarray import BaseDocument
            import numpy as np

            from docarray.typing import AudioUrl


            class MyAudio(Document):
                url: AudioUrl
                tensor: Optional[NdArray]
                bytes: Optional[bytes]


            doc = MyAudio(url="toydata/hello.wav")
            doc.bytes = doc.url.load_bytes()
            doc.tensor = doc.bytes.load()

            # Note this is equivalent to do

            doc.tensor = doc.url.load()

            assert isinstance(doc.audio_tensor, np.ndarray)

        :return: np.ndarray representing the Audio as RGB values
        """

        # note wave is Python built-in mod. https://docs.python.org/3/library/wave.html
        with wave.open(io.BytesIO(self)) as ifile:
            samples = ifile.getnframes()
            audio = ifile.readframes(samples)

            # Convert buffer to float32 using NumPy
            audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
            audio_as_np_float32 = audio_as_np_int16.astype('float32')

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
