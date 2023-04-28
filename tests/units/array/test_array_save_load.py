import os

import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray
    text: str
    image: ImageDoc


@pytest.mark.slow
@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle', 'json-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize('show_progress', [False, True])
def test_array_save_load_binary(protocol, compress, tmp_path, show_progress):
    tmp_file = os.path.join(tmp_path, 'test')

    da = DocList[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ]
    )

    da.save_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    da2 = DocList[MyDoc].load_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
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
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle', 'json-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize('show_progress', [False, True])
def test_array_save_load_binary_streaming(protocol, compress, tmp_path, show_progress):
    tmp_file = os.path.join(tmp_path, 'test')

    da = DocList[MyDoc]()

    def _extend_da(num_docs=100):
        for _ in range(num_docs):
            da.extend(
                [
                    MyDoc(
                        embedding=np.random.rand(3, 2),
                        text='hello',
                        image=ImageDoc(url='aux.png'),
                    ),
                ]
            )

    _extend_da()

    da.save_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    da2 = DocList[MyDoc]()
    da_generator = DocList[MyDoc].load_binary(
        tmp_file, protocol=protocol, compress=compress, show_progress=show_progress
    )

    for i, doc in enumerate(da_generator):
        assert doc.id == da[i].id
        assert doc.text == da[i].text
        assert doc.image.url == da[i].image.url
        da2.append(doc)

    assert len(da2) == 100
