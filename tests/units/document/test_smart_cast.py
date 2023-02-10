from typing import Optional

import numpy as np
import torch

from docarray import BaseDocument
from docarray.typing import AnyUrl, NdArray, TorchTensor


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
        url: AnyUrl
        title: str

    class B(BaseDocument):
        url: str

    a = A(url='file.png', title='hello')

    b = B.smart_parse_obj(a.dict())

    assert b.url == a.url


def test_casting_with_key_match_2():
    class A(BaseDocument):
        url: TorchTensor  # mistake done on purpose
        title: str

    class B(BaseDocument):
        url: str

    a = A(url=torch.zeros(5), title='hello')

    b = B.smart_parse_obj(a.dict())

    assert b.url == a.title


def test_optional_field():
    class A(BaseDocument):
        url: Optional[AnyUrl]
        tensor: Optional[NdArray]

    class B(BaseDocument):
        link: AnyUrl
        array: NdArray

    a = A(url='file.png', tensor=np.zeros(3))

    b = B.smart_parse_obj(a.dict())
    assert b.link == a.url
    assert (b.array == a.tensor).all()


def test_docstring():
    class A(BaseDocument):
        url_0: AnyUrl
        url: AnyUrl

        tensor: NdArray
        tensor_2: NdArray

    class B(BaseDocument):
        url: AnyUrl
        array: NdArray

    a = A(url='file.png', url_0='hello', tensor=np.zeros(3), tensor_2=np.zeros(4))
    b = B.smart_parse_obj(a.dict())
    assert b.url == a.url  # True
    assert (b.array == a.tensor).all()  # True


def test_cast_map():
    class A(BaseDocument):
        url_0: AnyUrl
        url: AnyUrl

        tensor: NdArray
        tensor_2: NdArray

    class B(BaseDocument):
        url: AnyUrl
        array: NdArray

    a = A(url='file.png', url_0='hello', tensor=np.zeros(3), tensor_2=np.zeros(4))
    b = B.smart_parse_obj(a.dict(), {'url': 'url_0'})
    assert b.url == a.url_0  # True
    assert (b.array == a.tensor).all()  # True
