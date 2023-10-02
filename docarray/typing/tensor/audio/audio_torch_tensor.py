from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor
from docarray.typing.tensor.torch_tensor import TorchTensor, metaTorchAndNode


@_register_proto(proto_type_name='audio_torch_tensor')
class AudioTorchTensor(AbstractAudioTensor, TorchTensor, metaclass=metaTorchAndNode):
    """
    Subclass of [`TorchTensor`][docarray.typing.TorchTensor], to represent an audio tensor.
    Adds audio-specific features to the tensor.

    ---

    ```python
    from typing import Optional

    import torch

    from docarray import BaseDoc
    from docarray.typing import AudioBytes, AudioTorchTensor, AudioUrl


    class MyAudioDoc(BaseDoc):
        title: str
        audio_tensor: Optional[AudioTorchTensor] = None
        url: Optional[AudioUrl] = None
        bytes_: Optional[AudioBytes] = None


    doc_1 = MyAudioDoc(
        title='my_first_audio_doc',
        audio_tensor=torch.zeros(1000, 2),
    )

    # doc_1.audio_tensor.save(file_path='/tmp/file_1.wav')
    doc_1.bytes_ = doc_1.audio_tensor.to_bytes()

    doc_2 = MyAudioDoc(
        title='my_second_audio_doc',
        url='https://www.kozco.com/tech/piano2.wav',
    )

    doc_2.audio_tensor, _ = doc_2.url.load()
    # doc_2.audio_tensor.save(file_path='/tmp/file_2.wav')
    doc_2.bytes_ = doc_1.audio_tensor.to_bytes()
    ```

    ---
    """

    ...
