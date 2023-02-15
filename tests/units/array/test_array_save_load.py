import pytest
import os
import numpy as np

from docarray import BaseDocument
from docarray.typing import NdArray
from docarray.documents import Image
from docarray import DocumentArray


class MyDoc(BaseDocument):
    embedding: NdArray
    text: str
    image: Image


@pytest.mark.slow
@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_array_save_load_binary(protocol, compress, tmp_path):
    tmp_file = os.path.join(tmp_path, 'test')

    da = DocumentArray[MyDoc](
        [
            MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=Image(url='aux.png')),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=Image()),
        ]
    )

    da.save_binary(tmp_file, protocol=protocol, compress=compress)

    da2 = DocumentArray[MyDoc].load_binary(
        tmp_file, protocol=protocol, compress=compress
    )

    assert len(da2) == 2
    assert len(da) == len(da2)
    for d1, d2 in zip(da, da2):
        assert d1.embedding.tolist() == d2.embedding.tolist()
        assert d1.text == d2.text
        assert d1.image.url == d2.image.url
    assert da[1].image.url is None
    assert da2[1].image.url is None


@pytest.mark.slow
@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_array_save_load_binary_streaming(protocol, compress, tmp_path):
    tmp_file = os.path.join(tmp_path, 'test')

    da = DocumentArray[MyDoc]()

    def _extend_da(num_docs=100):
        for _ in range(num_docs):
            da.extend(
                [
                    MyDoc(
                        embedding=np.random.rand(3, 2),
                        text='hello',
                        image=Image(url='aux.png'),
                    ),
                ]
            )

    _extend_da()

    da.save_binary(tmp_file, protocol=protocol, compress=compress)

    da2 = DocumentArray[MyDoc]()
    da_generator = DocumentArray[MyDoc].load_binary(
        tmp_file, protocol=protocol, compress=compress
    )

    for i, doc in enumerate(da_generator):
        assert doc.id == da[i].id
        assert doc.text == da[i].text
        assert doc.image.url == da[i].image.url
        da2.append(doc)

    assert len(da2) == 100
