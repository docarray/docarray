from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
from pydantic import Field

from docarray.base_doc import BaseDoc
from docarray.documents import AudioDoc
from docarray.typing import AnyEmbedding, AnyTensor, VideoBytes
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.video.video_tensor import VideoTensor
from docarray.typing.url.video_url import VideoUrl
from docarray.utils._internal.misc import import_library
from docarray.utils._internal.pydantic import is_pydantic_v2

if is_pydantic_v2:
    from pydantic import model_validator

if TYPE_CHECKING:
    import tensorflow as tf  # type: ignore
    import torch
else:
    tf = import_library('tensorflow', raise_error=False)
    torch = import_library('torch', raise_error=False)


T = TypeVar('T', bound='VideoDoc')


class VideoDoc(BaseDoc):
    """
    Document for handling video.

    The Video Document can contain:

    - a [`VideoUrl`][docarray.typing.url.VideoUrl] (`VideoDoc.url`)
    - an [`AudioDoc`][docarray.documents.AudioDoc] (`VideoDoc.audio`)
    - a [`VideoTensor`](../../../api_references/typing/tensor/video) (`VideoDoc.tensor`)
    - an [`AnyTensor`](../../../api_references/typing/tensor/tensor) representing the indices of the video's key frames (`VideoDoc.key_frame_indices`)
    - an [`AnyEmbedding`](../../../api_references/typing/tensor/embedding) (`VideoDoc.embedding`)
    - a [`VideoBytes`][docarray.typing.bytes.VideoBytes] object (`VideoDoc.bytes_`)

    You can use this Document directly:

    ```python
    from docarray.documents import VideoDoc, AudioDoc

    # use it directly
    vid = VideoDoc(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
    )
    tensor, audio_tensor, key_frame_indices = vid.url.load()
    vid.tensor = tensor
    vid.audio = AudioDoc(tensor=audio_tensor)
    vid.key_frame_indices = key_frame_indices
    # model = MyEmbeddingModel()
    # vid.embedding = model(vid.tensor)
    ```

    You can extend this Document:

    ```python
    from typing import Optional

    from docarray.documents import TextDoc, VideoDoc


    # extend it
    class MyVideo(VideoDoc):
        name: Optional[TextDoc] = None


    video = MyVideo(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
    )
    video.name = TextDoc(text='my first video')
    video.tensor = video.url.load().video
    # model = MyEmbeddingModel()
    # video.embedding = model(video.tensor)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import TextDoc, VideoDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        video: VideoDoc
        text: TextDoc


    mmdoc = MultiModalDoc(
        video=VideoDoc(
            url='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true'
        ),
        text=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.video.tensor = mmdoc.video.url.load().video

    # or
    mmdoc.video.bytes_ = mmdoc.video.url.load_bytes()
    mmdoc.video.tensor = mmdoc.video.bytes_.load().video
    ```
    """

    url: Optional[VideoUrl] = Field(
        description='URL to a (potentially remote) video file that needs to be loaded',
        example='https://github.com/docarray/docarray/blob/main/tests/toydata/mov_bbb.mp4?raw=true',
        default=None,
    )
    audio: Optional[AudioDoc] = Field(
        description='Audio document associated with the video',
        default=None,
    )
    tensor: Optional[VideoTensor] = Field(
        description='Tensor object representing the video which be specified to one of `VideoNdArray`, `VideoTorchTensor`, `VideoTensorFlowTensor`',
        default=None,
    )
    key_frame_indices: Optional[AnyTensor] = Field(
        description='List of all the key frames in the video',
        example=[0, 1, 2, 3, 4],
        default=None,
    )
    embedding: Optional[AnyEmbedding] = Field(
        description='Store an embedding: a vector representation of the video',
        example=[1, 0, 1],
        default=None,
    )
    bytes_: Optional[VideoBytes] = Field(
        description='Bytes representation of the video',
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
