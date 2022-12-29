import wave
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, BinaryIO, TypeVar, Union

from docarray.typing.tensor.abstract_tensor import AbstractTensor

T = TypeVar('T', bound='AbstractAudioTensor')

if TYPE_CHECKING:
    from docarray.proto import NodeProto


class AbstractAudioTensor(AbstractTensor, ABC):
    @abstractmethod
    def to_audio_bytes(self):
        """
        Convert audio tensor to bytes.
        """
        ...

    def _to_node_protobuf(self: T) -> 'NodeProto':
        """
        Convert itself into a NodeProto protobuf message. This function should
        be called when the Document is nested into another Document that need to be
        converted into a protobuf
        :param field: field in which to store the content in the node proto
        :return: the nested item protobuf message
        """
        from docarray.proto import NodeProto

        nd_proto = self.to_protobuf()
        return NodeProto(**{self.TENSOR_FIELD_NAME: nd_proto})

    def save_to_wav_file(
        self: 'T',
        file_path: Union[str, BinaryIO],
        sample_rate: int = 44100,
        sample_width: int = 2,
    ) -> None:
        """
        Save audio tensor to a .wav file. Mono/stereo is preserved.

        :param file_path: path to a .wav file. If file is a string, open the file by
            that name, otherwise treat it as a file-like object.
        :param sample_rate: sampling frequency
        :param sample_width: sample width in bytes
        """
        n_channels = 2 if self.n_dim() > 1 else 1

        with wave.open(file_path, 'w') as f:
            f.setnchannels(n_channels)
            f.setsampwidth(sample_width)
            f.setframerate(sample_rate)
            f.writeframes(self.to_audio_bytes())
