import io
from typing import TYPE_CHECKING, Any, Tuple, Type, TypeVar

import numpy as np
from pydantic import parse_obj_as
from pydantic.validators import bytes_validator

from docarray.typing.abstract_type import AbstractType
from docarray.typing.proto_register import _register_proto

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
    def doc_from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    def load(self) -> Tuple[np.ndarray, int]:
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
            doc.tensor, doc.frame_rate = doc.bytes.load()

            # Note this is equivalent to do

            doc.tensor, doc.frame_rate = doc.url.load()

            assert isinstance(doc.audio_tensor, np.ndarray)

        :return: np.ndarray representing the Audio as RGB values
        """

        from pydub import AudioSegment  # type: ignore

        segment = AudioSegment.from_file(io.BytesIO(self))

        # Convert to float32 using NumPy
        samples = np.array(segment.get_array_of_samples())

        # Normalise float32 array so that values are between -1.0 and +1.0
        samples_norm = samples / 2 ** (segment.sample_width * 8 - 1)
        return samples_norm, segment.frame_rate
