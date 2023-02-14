import pytest

from docarray import BaseDocument
from docarray.typing import NdArray
from docarray.documents import Image
from docarray import DocumentArray


class MyDoc(BaseDocument):
    embedding: NdArray
    text: str
    image: Image


def test_from_to_json():
    da = DocumentArray[MyDoc](
        [
            MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=Image(url='aux.png')),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=Image()),
        ]
    )
    json_da = da.to_json()
    da2 = DocumentArray[MyDoc].from_json(
        json_da
    )
    assert len(da2) == 2
    assert len(da) == len(da2)
    for d1, d2 in zip(da, da2):
        assert d1.embedding.tolist() == d2.embedding.tolist()
        assert d1.text == d2.text
        assert d1.image.url == d2.image.url
    assert da[1].image.url is None
    assert da2[1].image.url is None
