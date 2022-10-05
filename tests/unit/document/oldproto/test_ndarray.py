import numpy as np
import paddle
import pytest
import tensorflow as tf

from docarray import Document


def get_ndarrays():
    a = np.random.random([10, 3])
    a[a > 0.5] = 0
    return [
        (paddle.to_tensor(a), False),
        (tf.constant(a), False),
        (tf.sparse.from_dense(a), True),
    ]


@pytest.mark.parametrize('ndarray_val, is_sparse', get_ndarrays())
@pytest.mark.parametrize('attr', ['embedding', 'tensor'])
def test_ndarray_force_numpy(ndarray_val, attr, is_sparse):
    d = Document()
    setattr(d, attr, ndarray_val)
    assert type(getattr(Document.from_protobuf(d.to_protobuf()), attr)) is type(
        ndarray_val
    )
