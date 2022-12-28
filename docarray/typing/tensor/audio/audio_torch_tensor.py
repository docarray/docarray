import wave
from typing import TYPE_CHECKING, BinaryIO, TypeVar, Union

from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode

T = TypeVar('T', bound='AudioTorchTensor')

if TYPE_CHECKING:
    from docarray.proto import NodeProto


class AudioTorchTensor(TorchTensor, metaclass=metaTorchAndNode):
    """
    Subclass of TorchTensor, to represent an audio tensor.
    Additionally, this allows storing such a tensor as a .wav audio file.


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        import torch
        from pydantic import parse_obj_as

        from docarray import Document
        from docarray.typing import AudioTorchTensor, AudioUrl


        class MyAudioDoc(Document):
            title: str
            audio_tensor: Optional[AudioTorchTensor]
            url: Optional[AudioUrl]


        doc_1 = MyAudioDoc(
            title='my_first_audio_doc',
            audio_tensor=torch.randn(size=(1000, 2)),
        )

        doc_1.audio_tensor.save_audio_tensor_to_file(file_path='path/to/file_1.wav')


        doc_2 = MyAudioDoc(
            title='my_second_audio_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc_2.audio_tensor = parse_obj_as(AudioTorchTensor, doc_2.url.load())
        doc_2.audio_tensor.save_audio_tensor_to_file(file_path='path/to/file_2.wav')

    """

    def _to_node_protobuf(self: T, field: str = 'audio_torch_tensor') -> 'NodeProto':
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
        import torch

        max_int16 = 2**15
        tensor = self * max_int16
        tensor.to(dtype=torch.int16)
        n_channels = 2 if self.ndim > 1 else 1

        with wave.open(file_path, 'w') as f:
            f.setnchannels(n_channels)
            f.setsampwidth(sample_width)
            f.setframerate(sample_rate)
            f.writeframes(tensor.cpu().detach().numpy().tobytes())
