import numpy as np
import paddle
import pytest

from docarray.math.distance import cdist


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            np.array([[0, 27], [27, 0]]),
        ),
    ),
)
def test_seqeuclidean(x_mat, y_mat, result):
    np.testing.assert_almost_equal(
        cdist(x_mat, y_mat, metric='sqeuclidean'), result, decimal=3
    )


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            np.array([[0, 5.196], [5.196, 0]]),
        ),
    ),
)
def test_euclidean(x_mat, y_mat, result):
    np.testing.assert_almost_equal(
        cdist(x_mat, y_mat, metric='euclidean'), result, decimal=3
    )
