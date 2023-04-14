import io
from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar

import numpy as np
from pydantic import parse_obj_as
from pydantic.validators import bytes_validator

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio import AudioNdArray
from docarray.utils._internal.misc import import_library

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
            tensor: Optional[AudioNdArray]
            bytes_: Optional[AudioBytes]
            frame_rate: Optional[float]


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
