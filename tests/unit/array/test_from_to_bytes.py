import pytest

from docarray import DocumentArray


@pytest.mark.parametrize('protocol', ['protobuf', 0, 1, 2, 3, 4])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = DocumentArray.empty(1_000)
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(d)


@pytest.mark.parametrize('protocol', ['protobuf', 0, 1, 2, 3, 4])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_save_bytes(protocol, compress, tmpfile):
    d = DocumentArray.empty(1_000)
    d.save_binary(tmpfile, protocol=protocol, compress=compress)
    d.save_binary(str(tmpfile), protocol=protocol, compress=compress)

    with open(tmpfile, 'wb') as fp:
        d.save_binary(fp, protocol=protocol, compress=compress)

    d.load_binary(tmpfile, protocol=protocol, compress=compress)
    d.load_binary(str(tmpfile), protocol=protocol, compress=compress)
    with open(tmpfile, 'rb') as fp:
        d.load_binary(fp, protocol=protocol, compress=compress)
