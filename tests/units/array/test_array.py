import numpy as np

from docarray import Document, DocumentArray
from docarray.document import BaseDocument
from docarray.typing import NdArray


def test_document_array():
    class Text(Document):
        text: str

    da = DocumentArray([Text(text='hello') for _ in range(10)])

    assert len(da) == 10


def test_document_array_fixed_type():
    class Text(Document):
        text: str

    da = DocumentArray[Text]([Text(text='hello') for _ in range(10)])

    assert len(da) == 10


def test_get_bulk_attributes_function():
    class Mmdoc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da._get_array_attribute('tensor')

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da._get_array_attribute('text')

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_set_attributes():
    class InnerDoc(BaseDocument):
        text: str

    class Mmdoc(BaseDocument):
        inner: InnerDoc

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(inner=InnerDoc(text=f'hello{i}')) for i in range(N))
    )

    list_docs = [InnerDoc(text=f'hello{i}') for i in range(N)]
    da._set_array_attribute('inner', list_docs)

    for doc, list_doc in zip(da, list_docs):
        assert doc.inner is list_doc


def test_get_bulk_attributes():
    class Mmdoc(BaseDocument):
        text: str
        tensor: NdArray

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(text=f'hello{i}', tensor=np.zeros((3, 224, 224))) for i in range(N))
    )

    tensors = da.tensor

    assert len(tensors) == N
    for tensor in tensors:
        assert tensor.shape == (3, 224, 224)

    texts = da.text

    assert len(texts) == N
    for i, text in enumerate(texts):
        assert text == f'hello{i}'


def test_get_bulk_attributes_document():
    class InnerDoc(BaseDocument):
        text: str

    class Mmdoc(BaseDocument):
        inner: InnerDoc

    N = 10

    da = DocumentArray[Mmdoc](
        (Mmdoc(inner=InnerDoc(text=f'hello{i}')) for i in range(N))
    )

    assert isinstance(da.inner, DocumentArray)
