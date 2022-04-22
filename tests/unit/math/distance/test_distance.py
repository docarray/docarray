import numpy as np
import paddle
import pytest
import tensorflow as tf
import torch

from docarray.math.distance import cdist, pdist


def test_pdist():
    np.testing.assert_almost_equal(
        pdist(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'cosine'),
        cdist(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'cosine',
        ),
        decimal=3,
    )


def test_cdist_raise_error():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        cdist(x, y, 'cosine')


def test_not_supported_metric():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(NotImplementedError):
        cdist(x, y, 'fake_metric')


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[0, 27], [27, 0]]),
        ),
        (
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            np.array([[0, 27], [27, 0]]),
        ),
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
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
        (
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
        (
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype='float32'),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
    ),
)
def test_euclidean(x_mat, y_mat, result):
    np.testing.assert_almost_equal(
        cdist(x_mat, y_mat, metric='euclidean'), result, decimal=3
    )
