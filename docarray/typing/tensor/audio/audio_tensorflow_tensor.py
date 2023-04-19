from typing import TypeVar

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor
from docarray.typing.tensor.tensorflow_tensor import TensorFlowTensor, metaTensorFlow

T = TypeVar('T', bound='AudioTensorFlowTensor')


@_register_proto(proto_type_name='audio_tensorflow_tensor')
class AudioTensorFlowTensor(
    AbstractAudioTensor, TensorFlowTensor, metaclass=metaTensorFlow
):
    """
    Subclass of [`TensorFlowTensor`][docarray.typing.TensorFlowTensor],
    to represent an audio tensor. Adds audio-specific features to the tensor.

    ---

    ```python
    from typing import Optional

    import tensorflow as tf

    from docarray import BaseDoc
    from docarray.typing import AudioBytes, AudioTensorFlowTensor, AudioUrl


    class MyAudioDoc(BaseDoc):
        title: str
        audio_tensor: Optional[AudioTensorFlowTensor]
        url: Optional[AudioUrl]
        bytes_: Optional[AudioBytes]


    doc_1 = MyAudioDoc(
        title='my_first_audio_doc',
        audio_tensor=tf.random.normal((1000, 2)),
    )

    # doc_1.audio_tensor.save(file_path='file_1.wav')
    doc_1.bytes_ = doc_1.audio_tensor.to_bytes()

    doc_2 = MyAudioDoc(
        title='my_second_audio_doc',
        url='https://www.kozco.com/tech/piano2.wav',
    )

    doc_2.audio_tensor, _ = doc_2.url.load()
    doc_2.bytes_ = doc_1.audio_tensor.to_bytes()
    ```

    ---
    """

    ...
