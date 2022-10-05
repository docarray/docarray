import numpy as np
import paddle
import pytest
import tensorflow as tf

from docarray import DocumentArray


def get_ndarrays():
    a = np.random.random([10, 3])
    a[a > 0.5] = 0
    return [
        paddle.to_tensor(a),
        tf.constant(a),
    ]


@pytest.mark.parametrize('ndarray_val', get_ndarrays())
def test_diff_framework_match(ndarray_val):
    da = DocumentArray.empty(10)
    da.embeddings = ndarray_val
    da.match(da)
