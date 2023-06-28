from typing import Any, Optional, Type, TypeVar, Union

from docarray.base_doc import BaseDoc
from docarray.typing import TextUrl
from docarray.typing.tensor.embedding import AnyEmbedding

T = TypeVar('T', bound='TextDoc')


class TextDoc(BaseDoc):
    """
    Document for handling text.

    It can contain:

    - a [`TextUrl`][docarray.typing.url.TextUrl] (`TextDoc.url`)
    - a `str` (`TextDoc.text`)
    - an [`AnyEmbedding`](../../../api_references/typing/tensor/embedding) (`TextDoc.embedding`)
    - a `bytes` object (`TextDoc.bytes_`)

    You can use this Document directly:

    ```python
    from docarray.documents import TextDoc

    # use it directly
    txt_doc = TextDoc(url='https://www.gutenberg.org/files/1065/1065-0.txt')
    txt_doc.text = txt_doc.url.load()
    # model = MyEmbeddingModel()
    # txt_doc.embedding = model(txt_doc.text)
    ```

    You can initialize directly from a string:

    ```python
    from docarray.documents import TextDoc

    txt_doc = TextDoc('hello world')
    ```

    You can extend this Document:

    ```python
    from docarray.documents import TextDoc
    from docarray.typing import AnyEmbedding
    from typing import Optional


    # extend it
    class MyText(TextDoc):
        second_embedding: Optional[AnyEmbedding]


    txt_doc = MyText(url='https://www.gutenberg.org/files/1065/1065-0.txt')
    txt_doc.text = txt_doc.url.load()
    # model = MyEmbeddingModel()
    # txt_doc.embedding = model(txt_doc.text)
    # txt_doc.second_embedding = model(txt_doc.text)
    ```

    You can use this Document for composition:

    ```python
    from docarray import BaseDoc
    from docarray.documents import ImageDoc, TextDoc


    # compose it
    class MultiModalDoc(BaseDoc):
        image_doc: ImageDoc
        text_doc: TextDoc


    mmdoc = MultiModalDoc(
        image_doc=ImageDoc(
            url='https://github.com/docarray/docarray/blob/main/tests/toydata/image-data/apple.png?raw=true'
        ),
        text_doc=TextDoc(text='hello world, how are you doing?'),
    )
    mmdoc.image_doc.tensor = mmdoc.image_doc.url.load()

    # or
    mmdoc.image_doc.bytes_ = mmdoc.image_doc.url.load_bytes()
    mmdoc.image_doc.tensor = mmdoc.image_doc.bytes_.load()
    ```

    This Document can be compared against another Document of the same type or a string.
    When compared against another object of the same type, the pydantic BaseModel
    equality check will apply which checks the equality of every attribute,
    excluding `id`. When compared against a str, it will check the equality
    of the `text` attribute against the given string.

    ```python
    from docarray.documents import TextDoc

    doc = TextDoc(text='This is the main text', url='exampleurl.com/file')
    doc2 = TextDoc(text='This is the main text', url='exampleurl.com/file')

    doc == 'This is the main text'  # True
    doc == doc2  # True
    ```

    """

    text: Optional[str]
    url: Optional[TextUrl]
    embedding: Optional[AnyEmbedding]
    bytes_: Optional[bytes]

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
        This method makes `TextDoc` behave the same as an `str`.

        :param item: A string to be checked if is a substring of `text` attribute
        :return: A boolean determining the presence of `item` as a substring in `text`

        ```python
        from docarray.documents import TextDoc

        t = TextDoc(text='this is my text document')
        assert 'text' in t
        assert 'docarray' not in t
        ```
        """
        if self.text is not None:
            return self.text.__contains__(item)
        else:
            return False

    def _get_string_for_regex_filter(self):
        return self.text
