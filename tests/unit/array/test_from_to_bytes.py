import numpy as np
import pytest
import tensorflow as tf
import torch
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, csc_matrix

from docarray import DocumentArray
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
    'protocol', ['protobuf', 'protobuf-once', 'pickle', 'pickle-once']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(target_da, protocol, compress, ndarray_val, is_sparse):
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)

    target_da.embeddings = ndarray_val
    target_da.blobs = ndarray_val
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    print(protocol, compress, len(bstr))
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)

    np.testing.assert_almost_equal(
        to_numpy_array(target_da.embeddings), to_numpy_array(da2.embeddings)
    )
    np.testing.assert_almost_equal(
        to_numpy_array(target_da.blobs), to_numpy_array(da2.blobs)
    )


@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
@pytest.mark.parametrize(
    'protocol', ['protobuf', 'protobuf-once', 'pickle', 'pickle-once']
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


@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
def test_from_to_safe_list(target_da):
    DocumentArray.from_list_safe(target_da.to_list_safe())
