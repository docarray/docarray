from typing import Optional

from docarray.base_document import BaseDocument
from docarray.typing import AnyEmbedding, AnyTensor, ImageUrl


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
    """

    url: Optional[ImageUrl]
    tensor: Optional[AnyTensor]
    embedding: Optional[AnyEmbedding]
