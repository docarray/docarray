from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor
from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode


@_register_proto(proto_type_name='audio_torch_tensor')
class AudioTorchTensor(AbstractAudioTensor, TorchTensor, metaclass=metaTorchAndNode):
    """
    Subclass of TorchTensor, to represent an audio tensor.
    Adds audio-specific features to the tensor.


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        import torch

        from docarray import BaseDocument
        from docarray.typing import AudioTorchTensor, AudioUrl


        class MyAudioDoc(BaseDocument):
            title: str
            audio_tensor: Optional[AudioTorchTensor]
            url: Optional[AudioUrl]
            bytes_: Optional[bytes]


        doc_1 = MyAudioDoc(
            title='my_first_audio_doc',
            audio_tensor=torch.randn(size=(1000, 2)),
        )

        doc_1.audio_tensor.save_to_wav_file(file_path='path/to/file_1.wav')
        doc_1.bytes_ = doc_1.audio_tensor.to_bytes()

        doc_2 = MyAudioDoc(
            title='my_second_audio_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc_2.audio_tensor = doc_2.url.load()
        doc_2.audio_tensor.save_to_wav_file(file_path='path/to/file_2.wav')
        doc_2.bytes_ = doc_1.audio_tensor.to_bytes()
    """

    ...
