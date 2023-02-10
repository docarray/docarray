import numpy as np

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

    b = B.from_protobuf_w_casting(a.to_protobuf())

    assert b.link == a.url
    assert (b.array == a.tensor).all()
