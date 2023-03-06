import pytest

from docarray import BaseDocument
from docarray.documents import ImageDoc
from docarray.typing import NdArray


class MyDoc(BaseDocument):
    embedding: NdArray
    text: str
    image: ImageDoc


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png'))

    assert d.text == 'hello'
    assert d.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d.image.url == 'aux.png'
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    d2 = MyDoc.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.text == 'hello'
    assert d2.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d2.image.url == 'aux.png'


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_base64(protocol, compress):
    d = MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png'))

    assert d.text == 'hello'
    assert d.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d.image.url == 'aux.png'
    bstr = d.to_base64(protocol=protocol, compress=compress)
    d2 = MyDoc.from_base64(bstr, protocol=protocol, compress=compress)
    assert d2.text == 'hello'
    assert d2.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d2.image.url == 'aux.png'
