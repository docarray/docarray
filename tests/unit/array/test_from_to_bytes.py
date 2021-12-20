import pytest

from docarray import DocumentArray
from tests import random_docs


@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
@pytest.mark.parametrize('protocol', ['protobuf', 0, 1, 2, 3, 4])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(target_da, protocol, compress):
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)


@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
@pytest.mark.parametrize('protocol', ['protobuf', 0, 1, 2, 3, 4])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_save_bytes(target_da, protocol, compress, tmpfile):
    target_da.save_binary(tmpfile, protocol=protocol, compress=compress)
    target_da.save_binary(str(tmpfile), protocol=protocol, compress=compress)

    with open(tmpfile, 'wb') as fp:
        target_da.save_binary(fp, protocol=protocol, compress=compress)

    DocumentArray.load_binary(tmpfile, protocol=protocol, compress=compress)
    DocumentArray.load_binary(str(tmpfile), protocol=protocol, compress=compress)
    with open(tmpfile, 'rb') as fp:
        DocumentArray.load_binary(fp, protocol=protocol, compress=compress)
