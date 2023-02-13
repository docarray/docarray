import numpy as np
import pytest

from docarray import BaseDocument
from docarray.typing import AnyUrl, NdArray


def test_simple_casting_proto():
    class A(BaseDocument):
        url: AnyUrl
        tensor: NdArray

    class B(BaseDocument):
        link: AnyUrl
        array: NdArray

    a = A(url='file.png', tensor=np.zeros(3))

    b = B.from_protobuf_smart(a.to_protobuf())

    assert b.link == a.url
    assert (b.array == a.tensor).all()


def test_cast_map_proto():
    class A(BaseDocument):
        url_0: AnyUrl
        url: AnyUrl

        tensor: NdArray
        tensor_2: NdArray

    class B(BaseDocument):
        url: AnyUrl
        array: NdArray

    a = A(url='file.png', url_0='hello', tensor=np.zeros(3), tensor_2=np.zeros(4))
    b = B.from_protobuf_smart(a.to_protobuf(), {'url': 'url_0'})

    assert b.url == a.url_0  # True
    assert (b.array == a.tensor).all()  # True


def test_fail():
    class A(BaseDocument):
        url_0: AnyUrl

    class B(BaseDocument):
        url: AnyUrl
        array: NdArray

    a = A(url_0='file.png')
    with pytest.raises(ValueError):
        B.from_protobuf_smart(a.to_protobuf())
