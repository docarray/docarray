import numpy as np
import pytest
from scipy.sparse import csr_matrix

from docarray.math.distance.numpy import (
    cosine,
    euclidean,
    sparse_cosine,
    sparse_euclidean,
    sparse_sqeuclidean,
    sqeuclidean,
)


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array(
                [[0.00000000e00, 2.53681537e-02], [2.53681537e-02, 2.22044605e-16]]
            ),
        ),
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), np.array([[0]])),
        (np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0]])),
        (np.array([[1, 2, 3]]), np.array([[19, 53, 201]]), np.array([[0.06788693]])),
    ),
)
def test_cosine(x_mat, y_mat, result):
    assert cosine(x_mat, y_mat).all() == result.all()


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            csr_matrix([[1, 2, 3], [4, 5, 6]]),
            csr_matrix([[1, 2, 3], [4, 5, 6]]),
            np.array(
                [[0.00000000e00, 2.53681537e-02], [2.53681537e-02, 2.22044605e-16]]
            ),
        ),
        (csr_matrix([[1, 2, 3]]), csr_matrix([[1, 2, 3]]), np.array([[0]])),
        (csr_matrix([[0, 0, 0]]), csr_matrix([[0, 0, 0]]), np.array([[np.nan]])),
        (
            csr_matrix([[1, 2, 3]]),
            csr_matrix([[19, 53, 201]]),
            np.array([[0.06788693]]),
        ),
    ),
)
def test_sparse_cosine(x_mat, y_mat, result):
    assert sparse_cosine(x_mat, y_mat).all() == result.all()


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 27], [27, 0]]),
        ),
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), np.array([[0]])),
        (np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0]])),
        (np.array([[1, 2, 3]]), np.array([[19, 53, 201]]), np.array([[42129]])),
    ),
)
def test_sqeuclidean(x_mat, y_mat, result):
    assert sqeuclidean(x_mat, y_mat).all() == result.all()


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            csr_matrix([[1, 2, 3], [4, 5, 6]]),
            csr_matrix([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 27], [27, 0]]),
        ),
        (csr_matrix([[1, 2, 3]]), csr_matrix([[1, 2, 3]]), np.array([[0]])),
        (csr_matrix([[0, 0, 0]]), csr_matrix([[0, 0, 0]]), np.array([[0]])),
        (csr_matrix([[1, 2, 3]]), csr_matrix([[19, 53, 201]]), np.array([[42129]])),
    ),
)
def test_sparse_sqeuclidean(x_mat, y_mat, result):
    assert sparse_sqeuclidean(x_mat, y_mat).all() == result.all()


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), np.array([[0]])),
        (np.array([[0, 0, 0]]), np.array([[0, 0, 0]]), np.array([[0]])),
        (np.array([[1, 2, 3]]), np.array([[19, 53, 201]]), np.array([[205.2535018]])),
    ),
)
def test_euclidean(x_mat, y_mat, result):
    assert euclidean(x_mat, y_mat).all() == result.all()


@pytest.mark.parametrize(
    'x_mat, y_mat, result',
    (
        (
            csr_matrix([[1, 2, 3], [4, 5, 6]]),
            csr_matrix([[1, 2, 3], [4, 5, 6]]),
            np.array([[0, 5.19615242], [5.19615242, 0]]),
        ),
        (csr_matrix([[1, 2, 3]]), csr_matrix([[1, 2, 3]]), np.array([[0]])),
        (csr_matrix([[0, 0, 0]]), csr_matrix([[0, 0, 0]]), np.array([[0]])),
        (
            csr_matrix([[1, 2, 3]]),
            csr_matrix([[19, 53, 201]]),
            np.array([[205.2535018]]),
        ),
    ),
)
def test_sparse_euclidean(x_mat, y_mat, result):
    assert sparse_euclidean(x_mat, y_mat).all() == result.all()
