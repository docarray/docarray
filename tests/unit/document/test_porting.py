import pytest

from docarray import Document, DocumentArray
from tests import random_docs


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = Document(embedding=[1, 2, 3, 4, 5], text='hello')
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    d2 = Document.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.non_empty_fields == d.non_empty_fields


@pytest.mark.parametrize('target', [DocumentArray.empty(10), random_docs(10)])
def test_dict_json(target):
    for d in target:
        d1 = Document.from_dict(d.to_dict())
        d2 = Document.from_json(d.to_json())
        assert d1 == d2
