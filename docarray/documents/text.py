from typing import Any, Optional, Type, TypeVar, Union

from docarray.base_document import BaseDocument
from docarray.typing import TextUrl
from docarray.typing.tensor.embedding import AnyEmbedding

T = TypeVar('T', bound='TextDoc')


class TextDoc(BaseDocument):
    """
    Document for handling text.
    It can contain a TextUrl (`TextDoc.url`), a str (`TextDoc.text`),
    and an AnyEmbedding (`TextDoc.embedding`).

    EXAMPLE USAGE:

    You can use this Document directly:

    .. code-block:: python

        from docarray.documents import TextDoc

        # use it directly
        txt_doc = Text(url='http://www.jina.ai/')
        txt_doc.text = txt_doc.url.load()
        model = MyEmbeddingModel()
        txt_doc.embedding = model(txt_doc.text)

    You can initialize directly from a string:

    .. code-block:: python

        from docarray.documents import TextDoc

        txt_doc = Text('hello world')

    You can extend this Document:

    .. code-block:: python

        from docarray.documents import TextDoc
        from docarray.typing import AnyEmbedding
        from typing import Optional

        # extend it
        class MyText(Text):
            second_embedding: Optional[AnyEmbedding]


        txt_doc = MyText(url='http://www.jina.ai/')
        txt_doc.text = txt_doc.url.load()
        model = MyEmbeddingModel()
        txt_doc.embedding = model(txt_doc.text)
        txt_doc.second_embedding = model(txt_doc.text)


    You can use this Document for composition:

    .. code-block:: python

        from docarray import BaseDocument
        from docarray.documents import ImageDoc, TextDoc

        # compose it
        class MultiModalDoc(BaseDocument):
            image_doc: Image
            text_doc: Text


        mmdoc = MultiModalDoc(
            image_doc=Image(url="http://www.jina.ai/image.jpg"),
            text_doc=Text(text="hello world, how are you doing?"),
        )
        mmdoc.text_doc.text = mmdoc.text_doc.url.load()

        # or

        mmdoc.text_doc.bytes = mmdoc.text_doc.url.load_bytes()


    This Document can be compared against another Document of the same type or a string.
    When compared against another object of the same type, the pydantic BaseModel
    equality check will apply which checks the equality of every attribute,
    including `id`. When compared against a str, it will check the equality
    of the `text` attribute against the given string.

    .. code-block:: python

        from docarray.documents import TextDoc

        doc = Text(text='This is the main text', url='exampleurl.com')
        doc2 = Text(text='This is the main text', url='exampleurl.com')

        doc == 'This is the main text'  # True
        doc == doc2  # False, their ids are not equivalent
    """

    text: Optional[str] = None
    url: Optional[TextUrl] = None
    embedding: Optional[AnyEmbedding] = None
    bytes: Optional[bytes] = None

    def __init__(self, text: Optional[str] = None, **kwargs):
        if 'text' not in kwargs:
            kwargs['text'] = text
        super().__init__(**kwargs)

    @classmethod
    def validate(
        cls: Type[T],
        value: Union[str, Any],
    ) -> T:
        if isinstance(value, str):
            value = cls(text=value)
        return super().validate(value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.text == other
        else:
            # BaseModel has a default equality
            return super().__eq__(other)

    def __contains__(self, item: str) -> bool:
        """
        This method makes `Text` behave the same as an `str`.

        .. code-block:: python

            from docarray.documents import Text

            t = Text(text='this is my text document')
            assert 'text' in t
            assert 'docarray' not in t

        :param item: A string to be checked if is a substring of `text` attribute
        :return: A boolean determining the presence of `item` as a substring in `text`
        """
        if self.text is not None:
            return self.text.__contains__(item)
        else:
            return False

    def _get_string_for_regex_filter(self):
        return self.text
