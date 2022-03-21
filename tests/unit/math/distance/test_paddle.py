import numpy as np
import paddle
import pytest

from docarray.math.distance.paddle import cosine, euclidean, sqeuclidean


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            np.array([[1.192093e-07, 2.53681537e-02], [2.53681537e-02, 0]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            np.array([[1.192093e-07]]),
        ),
        (
            paddle.to_tensor([[0, 0, 0]], dtype='float32'),
            paddle.to_tensor([[0, 0, 0]], dtype='float32'),
            np.array([[1]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            paddle.to_tensor([[19, 53, 201]], dtype='float32'),
            np.array([[0.06788693]]),
        ),
    ),
)
def test_cosine(x_mat, y_mat, result):
    np.testing.assert_allclose(cosine(x_mat, y_mat), result, rtol=1e-5)


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            np.array([[0, 27], [27, 0]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            np.array([[0]]),
        ),
        (
            paddle.to_tensor([[0, 0, 0]], dtype='float32'),
            paddle.to_tensor([[0, 0, 0]], dtype='float32'),
            np.array([[0]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            paddle.to_tensor([[19, 53, 201]], dtype='float32'),
            np.array([[42129]]),
        ),
    ),
)
def test_sqeuclidean(x_mat, y_mat, result):
    np.testing.assert_allclose(sqeuclidean(x_mat, y_mat), result, rtol=1e-5)


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            np.array([[0]]),
        ),
        (
            paddle.to_tensor([[0, 0, 0]], dtype='float32'),
            paddle.to_tensor([[0, 0, 0]], dtype='float32'),
            np.array([[0]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3]], dtype='float32'),
            paddle.to_tensor([[19, 53, 201]], dtype='float32'),
            np.array([[205.2535018]]),
        ),
    ),
)
def test_euclidean(x_mat, y_mat, result):
    np.testing.assert_allclose(euclidean(x_mat, y_mat), result, rtol=1e-5)
