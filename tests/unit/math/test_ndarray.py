import numpy as np
import paddle
import pytest
import tensorflow as tf
import torch

from docarray.math.ndarray import get_array_rows
from scipy.sparse import csr_matrix, coo_matrix, bsr_matrix, csc_matrix, issparse


@pytest.mark.parametrize(
    'data, expected_result',
    [
        ([1, 2, 3], (1, 1)),
        ([[1, 2, 3]], (1, 2)),
        ([[1, 2], [3, 4]], (2, 2)),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], (4, 2)),
    ],
)
@pytest.mark.parametrize(
    'arraytype',
    [
        list,
        torch.tensor,
        tf.constant,
        paddle.to_tensor,
        torch.tensor,
        csr_matrix,
        bsr_matrix,
        coo_matrix,
        csc_matrix,
    ],
)
def test_get_array_rows(data, expected_result, arraytype):
    data_array = arraytype(data)

    num_rows, ndim = get_array_rows(data_array)
    if issparse(data_array):
        # there is no ndim==1 in scipy sparse matrices therefore don't check
        assert expected_result[0] == num_rows
    else:
        assert expected_result == (num_rows, ndim)
