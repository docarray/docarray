from typing import Any, Optional, Type, TypeVar, Union

import numpy as np

from docarray.base_document import BaseDocument
from docarray.typing import AnyEmbedding, ImageBytes, ImageUrl
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.typing.tensor.image.image_tensor import ImageTensor

T = TypeVar('T', bound='Image')

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


class Image(BaseDocument):
    """
    Document for handling images.
    It can contain an ImageUrl (`Image.url`), an AnyTensor (`Image.tensor`),
    and an AnyEmbedding (`Image.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import Image

        # use it directly
        image = Image(url='http://www.jina.ai/image.jpg')
        image.tensor = image.url.load()
        model = MyEmbeddingModel()
        image.embedding = model(image.tensor)

    You can extend this Document:

    .. code-block:: python

        from docarray.documents import Image
        from docarray.typing import AnyEmbedding
        from typing import Optional

        # extend it
        class MyImage(Image):
            second_embedding: Optional[AnyEmbedding]


        image = MyImage(url='http://www.jina.ai/image.jpg')
        image.tensor = image.url.load()
        model = MyEmbeddingModel()
        image.embedding = model(image.tensor)
        image.second_embedding = model(image.tensor)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.documents import Image, Text

        # compose it
        class MultiModalDoc(BaseDocument):
            image: Image
            text: Text


        mmdoc = MultiModalDoc(
            image=Image(url="http://www.jina.ai/image.jpg"),
            text=Text(text="hello world, how are you doing?"),
        )
        mmdoc.image.tensor = mmdoc.image.url.load()
        # or
        mmdoc.image.bytes = mmdoc.image.url.load_bytes()

        mmdoc.image.tensor = mmdoc.image.bytes.load()
    """

    url: Optional[ImageUrl]
    tensor: Optional[ImageTensor]
    embedding: Optional[AnyEmbedding]
    bytes: Optional[ImageBytes]

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, AbstractTensor, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(url=value)
        elif isinstance(value, (AbstractTensor, np.ndarray)) or (
            torch_available and isinstance(value, torch.Tensor)
        ):
            value = cls(tensor=value)
        elif isinstance(value, bytes):
            value = cls(byte=value)

        return super().validate(value)
