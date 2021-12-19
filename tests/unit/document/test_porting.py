import pytest

from docarray import Document


@pytest.mark.parametrize('protocol', ['protobuf', 0, 1, 2, 3, 4])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = Document(embedding=[1, 2, 3, 4, 5], text='hello')
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    d2 = Document.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.non_empty_fields == d.non_empty_fields
