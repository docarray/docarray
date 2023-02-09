import numpy as np

from docarray import BaseDocument
from docarray.typing import AnyUrl, NdArray


def test_simple_casting():
    class A(BaseDocument):
        url: AnyUrl
        tensor: NdArray

    class B(BaseDocument):
        link: AnyUrl
        array: NdArray

    a = A(url='file.png', tensor=np.zeros(3))

    b = B.smart_parse_obj(a.dict())
    assert b.link == a.url
    assert (b.array == a.tensor).all()


def test_casting_with_key_match():
    class A(BaseDocument):
        ch: AnyUrl
        title: str

    class B(BaseDocument):
        url: str

    a = A(url='file.png', title='hello')

    b = B.smart_parse_obj(a.dict())

    assert b.url == a.title
