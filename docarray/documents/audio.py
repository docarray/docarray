from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_doc import BaseDoc
from docarray.typing import AnyEmbedding, AudioUrl
from docarray.typing.bytes.audio_bytes import AudioBytes
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.audio.audio_tensor import AudioTensor
from docarray.utils._internal.misc import import_library

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
        name: Optional[TextDoc]


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

    url: Optional[AudioUrl]
    tensor: Optional[AudioTensor]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[AudioBytes]
    frame_rate: Optional[int]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(url=value)
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch is not None
            and isinstance(value, torch.Tensor)
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = cls(tensor=value)

        return super().validate(value)
