from typing import Optional

from docarray.document import BaseDocument
from docarray.typing import TextUrl
from docarray.typing.tensor.embedding import Embedding


class Text(BaseDocument):
    """
    Document for handling text.
    It can contain a TextUrl (`Text.url`), a str (`Text.text`),
    and an Embedding (`Text.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray import Text

        # use it directly
        txt_doc = Text(url='http://www.jina.ai/')
        txt_doc.text = txt_doc.url.load()
        model = MyEmbeddingModel()
        txt_doc.embedding = model(txt_doc.text)

    You can extend this Document:

    .. code-block:: python

        from docarray import Text
        from docarray.typing import Embedding
        from typing import Optional

        # extend it
        class MyText(Text):
            second_embedding: Optional[Embedding]


        txt_doc = MyText(url='http://www.jina.ai/')
        txt_doc.text = txt_doc.url.load()
        model = MyEmbeddingModel()
        txt_doc.embedding = model(txt_doc.text)
        txt_doc.second_embedding = model(txt_doc.text)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import Document, Image, Text

        # compose it
        class MultiModalDoc(Document):
            image_doc: Image
            text_doc: Text


        mmdoc = MultiModalDoc(
            image_doc=Image(url="http://www.jina.ai/image.jpg"),
            text_doc=Text(text="hello world, how are you doing?"),
        )
        mmdoc.text_doc.text = mmdoc.text_doc.url.load()
    """

    text: Optional[str] = None
    url: Optional[TextUrl] = None
    embedding: Optional[Embedding] = None
