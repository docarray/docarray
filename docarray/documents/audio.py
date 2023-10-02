from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
from pydantic import Field

from docarray.base_doc import BaseDoc
from docarray.typing import AnyEmbedding, AudioUrl
from docarray.typing.bytes.audio_bytes import AudioBytes
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.audio.audio_tensor import AudioTensor
from docarray.utils._internal.misc import import_library
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic import model_validator

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
else:
    torch = import_library('torch', raise_error=False)
    tf = import_library('tensorflow', raise_error=False)


T = TypeVar('T', bound='AudioDoc')


class AudioDoc(BaseDoc):
    """
    Document for handling audios.

    The Audio Document can contain:

    - an [`AudioUrl`][docarray.typing.url.AudioUrl] (`AudioDoc.url`)
    - an [`AudioTensor`](../../../api_references/typing/tensor/audio) (`AudioDoc.tensor`)
    - an [`AnyEmbedding`](../../../api_references/typing/tensor/embedding) (`AudioDoc.embedding`)
    - an [`AudioBytes`][docarray.typing.bytes.AudioBytes] (`AudioDoc.bytes_`) object
    - an integer representing the frame_rate (`AudioDoc.frame_rate`)

    You can use this Document directly:

    ```python
    from docarray.documents import AudioDoc

    # use it directly
    audio = AudioDoc(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/hello.wav?raw=true'
    )
    audio.tensor, audio.frame_rate = audio.url.load()
    # model = MyEmbeddingModel()
    # audio.embedding = model(audio.tensor)
    ```

    You can extend this Document:

    ```python
    from docarray.documents import AudioDoc, TextDoc
    from typing import Optional


    # extend it
    class MyAudio(AudioDoc):
        name: Optional[TextDoc] = None


    audio = MyAudio(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/hello.wav?raw=true'
    )
    audio.name = TextDoc(text='my first audio')
    audio.tensor, audio.frame_rate = audio.url.load()
    # model = MyEmbeddingModel()
    # audio.embedding = model(audio.tensor)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import AudioDoc, TextDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        audio: AudioDoc
        text: TextDoc


    mmdoc = MultiModalDoc(
        audio=AudioDoc(
            url='https://github.com/docarray/docarray/blob/main/tests/toydata/hello.wav?raw=true'
        ),
        text=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.audio.tensor, mmdoc.audio.frame_rate = mmdoc.audio.url.load()

    # equivalent to
    mmdoc.audio.bytes_ = mmdoc.audio.url.load_bytes()
    mmdoc.audio.tensor, mmdoc.audio.frame_rate = mmdoc.audio.bytes_.load()
    ```
    """

    url: Optional[AudioUrl] = Field(
        description='The url to a (potentially remote) audio file that can be loaded',
        example='https://github.com/docarray/docarray/blob/main/tests/toydata/hello.mp3?raw=true',
        default=None,
    )
    tensor: Optional[AudioTensor] = Field(
        description='Tensor object of the audio which can be specified to one of `AudioNdArray`, `AudioTorchTensor`, `AudioTensorFlowTensor`',
        default=None,
    )
    embedding: Optional[AnyEmbedding] = Field(
        description='Store an embedding: a vector representation of the audio.',
        example=[0, 1, 0],
        default=None,
    )
    bytes_: Optional[AudioBytes] = Field(
        description='Bytes representation pf the audio',
        default=None,
    )
    frame_rate: Optional[int] = Field(
        description='An integer representing the frame rate of the audio.',
        example=24,
        default=None,
    )

    @classmethod
    def _validate(cls, value) -> Dict[str, Any]:
        if isinstance(value, str):
            value = dict(url=value)
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch is not None
            and isinstance(value, torch.Tensor)
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = dict(tensor=value)

        return value

    if is_pydantic_v2:

        @model_validator(mode='before')
        @classmethod
        def validate_model_before(cls, value):
            return cls._validate(value)

    else:

        @classmethod
        def validate(
            cls: Type[T],
            value: Union[str, AbstractTensor, Any],
        ) -> T:
            return super().validate(cls._validate(value))
