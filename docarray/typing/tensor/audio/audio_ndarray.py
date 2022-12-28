import wave
from typing import TYPE_CHECKING, BinaryIO, TypeVar, Union

from docarray.typing import NdArray

T = TypeVar('T', bound='AudioNdArray')

if TYPE_CHECKING:
    from docarray.proto import NodeProto


class AudioNdArray(NdArray):
    """ """

    def _to_node_protobuf(self: T, field: str = 'ndarray') -> 'NodeProto':
        """Convert itself into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        nd_proto = self.to_protobuf()
        return NodeProto(**{field: nd_proto})

    def save_audio_tensor_to_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        sample_rate: int = 44100,
        sample_width: int = 2,
    ) -> None:
        """
        Save :attr:`.tensor` into a .wav file. Mono/stereo is preserved.

        :param file_path: if file is a string, open the file by that name, otherwise
            treat it as a file-like object.
        :param sample_rate: sampling frequency
        :param sample_width: sample width in bytes
        """

        max_int16 = 2**15
        tensor = (self * max_int16).astype('<h')
        n_channels = 2 if self.ndim > 1 else 1

        with wave.open(file_path, 'w') as f:
            f.setnchannels(n_channels)
            f.setsampwidth(sample_width)
            f.setframerate(sample_rate)
            f.writeframes(tensor.tobytes())
