from typing import TYPE_CHECKING, Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
from pydantic import Field

from docarray.base_doc import BaseDoc
from docarray.typing import AnyEmbedding, ImageBytes, ImageUrl
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.image.image_tensor import ImageTensor
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

T = TypeVar('T', bound='ImageDoc')


class ImageDoc(BaseDoc):
    """
    Document for handling images.

    It can contain:

    - an [`ImageUrl`][docarray.typing.url.ImageUrl] (`Image.url`)
    - an [`ImageTensor`](../../../api_references/typing/tensor/image) (`Image.tensor`)
    - an [`AnyEmbedding`](../../../api_references/typing/tensor/embedding) (`Image.embedding`)
    - an [`ImageBytes`][docarray.typing.bytes.ImageBytes] object (`ImageDoc.bytes_`)

    You can use this Document directly:

    ```python
    from docarray.documents import ImageDoc

    # use it directly
    image = ImageDoc(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true'
    )
    image.tensor = image.url.load()
    # model = MyEmbeddingModel()
    # image.embedding = model(image.tensor)
    ```

    You can extend this Document:

    ```python
    from docarray.documents import ImageDoc
    from docarray.typing import AnyEmbedding
    from typing import Optional


    # extend it
    class MyImage(ImageDoc):
        second_embedding: Optional[AnyEmbedding] = None


    image = MyImage(
        url='https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true'
    )
    image.tensor = image.url.load()
    # model = MyEmbeddingModel()
    # image.embedding = model(image.tensor)
    # image.second_embedding = model(image.tensor)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import ImageDoc, TextDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        image: ImageDoc
        text: TextDoc


    mmdoc = MultiModalDoc(
        image=ImageDoc(
            url='https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true'
        ),
        text=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.image.tensor = mmdoc.image.url.load()

    # or
    mmdoc.image.bytes_ = mmdoc.image.url.load_bytes()
    mmdoc.image.tensor = mmdoc.image.bytes_.load()
    ```
    """

    url: Optional[ImageUrl] = Field(
        description='URL to a (potentially remote) image file that needs to be loaded',
        example='https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true',
        default=None,
    )
    tensor: Optional[ImageTensor] = Field(
        description='Tensor object of the image which can be specifed to one of `ImageNdArray`, `ImageTorchTensor`, `ImageTensorflowTensor`.',
        default=None,
    )
    embedding: Optional[AnyEmbedding] = Field(
        description='Store an embedding: a vector representation of the image.',
        example=[1, 0, 1],
        default=None,
    )
    bytes_: Optional[ImageBytes] = Field(
        description='Bytes object of the image which is an instance of `ImageBytes`.',
        default=None,
    )

    @classmethod
    def _validate(cls, value) -> Dict[str, Any]:
        if isinstance(value, str):
            value = dict(url=value)
        elif (
            isinstance(value, (AbstractTensor, np.ndarray))
            or (torch is not None and isinstance(value, torch.Tensor))
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = dict(tensor=value)
        elif isinstance(value, bytes):
            value = dict(byte=value)

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
