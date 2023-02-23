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
    Subclass of TensorFlowTensor, to represent an audio tensor.
    Adds audio-specific features to the tensor.


    EXAMPLE USAGE

    .. code-block:: python

        from typing import Optional

        import tensorflow as tf
        from pydantic import parse_obj_as

        from docarray import BaseDocument
        from docarray.typing import AudioTensorFlowTensor, AudioUrl


        class MyAudioDoc(BaseDocument):
            title: str
            audio_tensor: Optional[AudioTensorFlowTensor]
            url: Optional[AudioUrl]
            bytes_: Optional[bytes]


        doc_1 = MyAudioDoc(
            title='my_first_audio_doc',
            audio_tensor=tf.random.normal((1000, 2)),
        )

        doc_1.audio_tensor.save(file_path='path/to/file_1.wav')
        doc_1.bytes_ = doc_1.audio_tensor.to_bytes()


        doc_2 = MyAudioDoc(
            title='my_second_audio_doc',
            url='https://www.kozco.com/tech/piano2.wav',
        )

        doc_2.audio_tensor = doc_2.url.load()
        doc_2.audio_tensor.save(file_path='path/to/file_2.wav')
        doc_2.bytes_ = doc_1.audio_tensor.to_bytes()

    """

    ...
