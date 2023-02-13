import pytest

from docarray import BaseDocument
from docarray.typing import NdArray
from docarray.documents import Image


class MyDoc(BaseDocument):
    embedding: NdArray
    text: str
    image: Image


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=Image(url='aux.png'))

    assert d.text == 'hello'
    assert d.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d.image.url == 'aux.png'
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    d2 = MyDoc.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.text == 'hello'
    assert d2.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d2.image.url == 'aux.png'
