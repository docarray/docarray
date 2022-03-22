import numpy as np
import pytest
import tensorflow as tf

from docarray.math.distance.tensorflow import cosine, euclidean, sqeuclidean


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            np.array([[1.192093e-07, 2.53681537e-02], [2.53681537e-02, 0.000000e00]]),
        ),
        (
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            np.array([[1.192093e-07]]),
        ),
        (
            tf.constant([[0, 0, 0]], dtype=tf.float32),
            tf.constant([[0, 0, 0]], dtype=tf.float32),
            np.array([[1]]),
        ),
        (
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            tf.constant([[19, 53, 201]], dtype=tf.float32),
            np.array([[0.06788693]]),
        ),
    ),
)
def test_cosine(x_mat, y_mat, result):
    np.testing.assert_almost_equal(cosine(x_mat, y_mat), result, decimal=3)


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            np.array([[0, 27], [27, 0]]),
        ),
        (
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            np.array([[0]]),
        ),
        (
            tf.constant([[0, 0, 0]], dtype=tf.float32),
            tf.constant([[0, 0, 0]], dtype=tf.float32),
            np.array([[0]]),
        ),
        (
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            tf.constant([[19, 53, 201]], dtype=tf.float32),
            np.array([[42129]]),
        ),
    ),
)
def test_sqeuclidean(x_mat, y_mat, result):
    np.testing.assert_almost_equal(sqeuclidean(x_mat, y_mat), result, decimal=3)


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
        (
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            np.array([[0]]),
        ),
        (
            tf.constant([[0, 0, 0]], dtype=tf.float32),
            tf.constant([[0, 0, 0]], dtype=tf.float32),
            np.array([[0]]),
        ),
        (
            tf.constant([[1, 2, 3]], dtype=tf.float32),
            tf.constant([[19, 53, 201]], dtype=tf.float32),
            np.array([[205.2535018]]),
        ),
    ),
)
def test_euclidean(x_mat, y_mat, result):
    np.testing.assert_almost_equal(euclidean(x_mat, y_mat), result, decimal=3)
