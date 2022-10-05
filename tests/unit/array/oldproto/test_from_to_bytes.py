import numpy as np
import pytest
import tensorflow as tf

from docarray import DocumentArray
from docarray.math.ndarray import to_numpy_array
from tests import random_docs


def get_ndarrays_for_ravel():
    a = np.random.random([100, 3])
    a[a > 0.5] = 0
    return [(tf.constant(a), False)]


@pytest.mark.parametrize('ndarray_val, is_sparse', get_ndarrays_for_ravel())
@pytest.mark.parametrize('target_da', [DocumentArray.empty(100), random_docs(100)])
@pytest.mark.parametrize(
    'protocol', ['protobuf', 'protobuf-array', 'pickle', 'pickle-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(target_da, protocol, compress, ndarray_val, is_sparse):
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)

    target_da.embeddings = ndarray_val
    target_da.tensors = ndarray_val
    bstr = target_da.to_bytes(protocol=protocol, compress=compress)
    da2 = DocumentArray.from_bytes(bstr, protocol=protocol, compress=compress)
    assert len(da2) == len(target_da)

    np.testing.assert_almost_equal(
        to_numpy_array(target_da.embeddings), to_numpy_array(da2.embeddings)
    )
    np.testing.assert_almost_equal(
        to_numpy_array(target_da.tensors), to_numpy_array(da2.tensors)
    )
