from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_doc import BaseDoc
from docarray.typing import AnyEmbedding, ImageBytes, ImageUrl
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.image.image_tensor import ImageTensor
from docarray.utils._internal.misc import import_library

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
        second_embedding: Optional[AnyEmbedding]


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

    url: Optional[ImageUrl]
    tensor: Optional[ImageTensor]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[ImageBytes]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(url=value)
        elif (
            isinstance(value, (AbstractTensor, np.ndarray))
            or (torch is not None and isinstance(value, torch.Tensor))
            or (tf is not None and isinstance(value, tf.Tensor))
        ):
            value = cls(tensor=value)
        elif isinstance(value, bytes):
            value = cls(byte=value)

        return super().validate(value)
