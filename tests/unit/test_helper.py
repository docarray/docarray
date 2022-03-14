import os
import pathlib

import pytest

from docarray.helper import (
    protocol_and_compress_from_file_path,
    add_protocol_and_compress_to_file_path,
    get_full_version,
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


@pytest.mark.parametrize('file_path', ['doc_array', './some_folder/doc_array'])
@pytest.mark.parametrize(
    'protocol', ['protobuf', 'protobuf-array', 'pickle', 'pickle-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip'])
def test_add_protocol_and_compress_to_file_path(file_path, compress, protocol):
    file_path_extended = add_protocol_and_compress_to_file_path(
        file_path, compress, protocol
    )
    file_path_suffixes = [
        e.replace('.', '') for e in pathlib.Path(file_path_extended).suffixes
    ]

    if compress:
        assert compress in file_path_suffixes
    if protocol:
        assert protocol in file_path_suffixes


def test_ci_vendor():
    if 'GITHUB_WORKFLOW' in os.environ:
        assert get_full_version()['ci-vendor'] == 'GitHub Actions'
