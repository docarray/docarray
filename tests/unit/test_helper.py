import pytest

from docarray.helper import (
    protocol_and_compress_from_file_path,
)


@pytest.mark.parametrize(
    'file_path', ['doc_array', '../docarray', './a_folder/docarray']
)
@pytest.mark.parametrize(
    'protocol', ['protobuf', 'protobuf-array', 'pickle', 'pickle-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_protocol_and_compress_from_file_path(file_path, protocol, compress):

    file_path_extended = file_path
    if protocol:
        file_path_extended += '.' + protocol
    if compress:
        file_path_extended += '.' + compress

    _protocol, _compress = protocol_and_compress_from_file_path(file_path_extended)

    assert _protocol in {'protobuf', 'protobuf-array', 'pickle', 'pickle-array', None}
    assert _compress in {'lz4', 'bz2', 'lzma', 'zlib', 'gzip', None}

    assert protocol == _protocol
    assert compress == _compress
