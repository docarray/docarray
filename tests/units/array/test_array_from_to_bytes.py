import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray
    text: str
    image: ImageDoc


@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize('show_progress', [False, True])
def test_from_to_bytes(protocol, compress, show_progress):
    da = DocList[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ]
    )
    bytes_da = da.to_bytes(
        protocol=protocol, compress=compress, show_progress=show_progress
    )
    da2 = DocList[MyDoc].from_bytes(
        bytes_da, protocol=protocol, compress=compress, show_progress=show_progress
    )
    assert len(da2) == 2
    assert len(da) == len(da2)
    for d1, d2 in zip(da, da2):
        assert d1.embedding.tolist() == d2.embedding.tolist()
        assert d1.text == d2.text
        assert d1.image.url == d2.image.url
    assert da[1].image.url is None
    assert da2[1].image.url is None


@pytest.mark.parametrize(
    'protocol', ['pickle-array', 'protobuf-array', 'protobuf', 'pickle']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize('show_progress', [False, True])
def test_from_to_base64(protocol, compress, show_progress):
    da = DocList[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ]
    )
    bytes_da = da.to_base64(
        protocol=protocol, compress=compress, show_progress=show_progress
    )
    da2 = DocList[MyDoc].from_base64(
        bytes_da, protocol=protocol, compress=compress, show_progress=show_progress
    )
    assert len(da2) == 2
    assert len(da) == len(da2)
    for d1, d2 in zip(da, da2):
        assert d1.embedding.tolist() == d2.embedding.tolist()
        assert d1.text == d2.text
        assert d1.image.url == d2.image.url
    assert da[1].image.url is None
    assert da2[1].image.url is None


def test_union_type_error(tmp_path):
    from typing import Union

    from docarray.documents import TextDoc

    class CustomDoc(BaseDoc):
        ud: Union[TextDoc, ImageDoc] = TextDoc(text='union type')

    docs = DocList[CustomDoc]([CustomDoc(ud=TextDoc(text='union type'))])

    with pytest.raises(ValueError):
        docs.from_bytes(docs.to_bytes())
