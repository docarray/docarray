from typing import TypeVar

from docarray.typing.proto_register import _register_proto
from docarray.typing.tensor.audio.abstract_audio_tensor import AbstractAudioTensor
from docarray.typing.tensor.jaxarray import JaxArray, metaJax

T = TypeVar('T', bound='AudioJaxArray')


@_register_proto(proto_type_name='audio_jaxarray')
class AudioJaxArray(AbstractAudioTensor, JaxArray, metaclass=metaJax):
    ...
