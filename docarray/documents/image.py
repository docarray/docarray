from typing import Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_doc import BaseDoc
from docarray.typing import AnyEmbedding, ImageBytes, ImageUrl
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.image.image_tensor import ImageTensor
from docarray.utils.misc import is_tf_available, is_torch_available

T = TypeVar('T', bound='ImageDoc')

torch_available = is_torch_available()
if torch_available:
    import torch

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf  # type: ignore


class ImageDoc(BaseDoc):
    """
    Document for handling images.
    It can contain an ImageUrl (`Image.url`), an AnyTensor (`Image.tensor`),
    and an AnyEmbedding (`Image.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import ImageDoc

        # use it directly
        image = ImageDoc(url='http://www.jina.ai/image.jpg')
        image.tensor = image.url.load()
        model = MyEmbeddingModel()
        image.embedding = model(image.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray.documents import ImageDoc
        from docarray.typing import AnyEmbedding
        from typing import Optional


        # extend it
        class MyImage(ImageDoc):
            second_embedding: Optional[AnyEmbedding]


        image = MyImage(url='http://www.jina.ai/image.jpg')
        image.tensor = image.url.load()
        model = MyEmbeddingModel()
        image.embedding = model(image.tensor)
        image.second_embedding = model(image.tensor)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import BaseDoc
        from docarray.documents import ImageDoc, TextDoc


        # compose it
        class MultiModalDoc(BaseDoc):
            image: Image
            text: Text


        mmdoc = MultiModalDoc(
            image=Image(url="http://www.jina.ai/image.jpg"),
            text=Text(text="hello world, how are you doing?"),
        )
        mmdoc.image.tensor = mmdoc.image.url.load()
        # or
        mmdoc.image.bytes_ = mmdoc.image.url.load_bytes()

        mmdoc.image.tensor = mmdoc.image.bytes.load()
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
            or (torch_available and isinstance(value, torch.Tensor))
            or (tf_available and isinstance(value, tf.Tensor))
        ):
            value = cls(tensor=value)
        elif isinstance(value, bytes):
            value = cls(byte=value)

        return super().validate(value)
