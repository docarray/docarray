import types

import numpy as np
import pytest
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, csc_matrix

from docarray import DocumentArray, Document
from docarray.math.ndarray import to_numpy_array
from tests import random_docs


def get_ndarrays_for_ravel():
    a = np.random.random([100, 3])
    a[a > 0.5] = 0
    return [
        (a, False),
        (torch.tensor(a), False),
        (tf.constant(a), False),
        (torch.tensor(a).to_sparse(), True),
        # (tf.sparse.from_dense(a), True),
        (csr_matrix(a), True),
        (bsr_matrix(a), True),
        (coo_matrix(a), True),
        (csc_matrix(a), True),
    ]


@pytest.mark.parametrize('ndarray_val, is_sparse', get_ndarrays_for_ravel())
@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
@pytest.mark.parametrize(
    'protocol', ['protobuf', 'protobuf-array', 'pickle', 'pickle-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(target_da, protocol, compress, ndarray_val, is_sparse):
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)

    target_da.embeddings = ndarray_val
    target_da.tensors = ndarray_val
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)

    np.testing.assert_almost_equal(
        to_numpy_array(target_da.embeddings), to_numpy_array(da2.embeddings)
    )
    np.testing.assert_almost_equal(
        to_numpy_array(target_da.tensors), to_numpy_array(da2.tensors)
    )


@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
@pytest.mark.parametrize(
    'protocol', ['protobuf', 'protobuf-array', 'pickle', 'pickle-array']
)
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


@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
def test_from_to_protobuf(target_da):
    DocumentArray.from_protobuf(target_da.to_protobuf())


@pytest.mark.parametrize('target', [DocumentArray.empty(10), random_docs(10)])
@pytest.mark.parametrize('protocol', ['jsonschema', 'protobuf'])
@pytest.mark.parametrize('to_fn', ['dict', 'json'])
def test_from_to_safe_list(target, protocol, to_fn):
    da_r = getattr(DocumentArray, f'from_{to_fn}')(
        getattr(target, f'to_{to_fn}')(protocol=protocol), protocol=protocol
    )
    assert da_r == target


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('show_progress', [True, False])
def test_push_pull_show_progress(show_progress, protocol):
    da = DocumentArray.empty(1000)
    r = da.to_bytes(_show_progress=show_progress, protocol=protocol)
    da_r = DocumentArray.from_bytes(r, _show_progress=show_progress, protocol=protocol)
    assert da == da_r


# Note  protocol = ['protobuf-array', 'pickle-array'] not supported with Document.from_bytes
@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize(
    'compress', ['lz4', 'bz2', 'lzma', 'gzip', 'zlib', 'gzib', None]
)
def test_save_bytes_stream(tmpfile, protocol, compress):
    da = DocumentArray(
        [Document(text='aaa'), Document(text='bbb'), Document(text='ccc')]
    )
    da.save_binary(tmpfile, protocol=protocol, compress=compress)

    # note that save_binary will save protocol and compression in the filename
    if protocol:
        tmpfile += '.' + protocol
    if compress:
        tmpfile += '.' + compress

    da_reconstructed = DocumentArray.load_binary(
        tmpfile, protocol=protocol, compress=compress, streaming=True
    )
    assert isinstance(da_reconstructed, types.GeneratorType)
    for d, d_rec in zip(da, da_reconstructed):
        assert d == d_rec
