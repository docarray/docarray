import numpy as np
import pytest

from docarray.computation.numpy_backend import NumpyCompBackend


def test_to_device():
    with pytest.raises(NotImplementedError):
        NumpyCompBackend.to_device(np.random.rand(10, 3), 'meta')


@pytest.mark.parametrize(
    'array,result',
    [
        (np.zeros((5)), 1),
        (np.zeros((1, 5)), 2),
        (np.zeros((5, 5)), 2),
        (np.zeros(()), 0),
    ],
)
def test_n_dim(array, result):
    assert NumpyCompBackend.n_dim(array) == result


@pytest.mark.parametrize(
    'array,result',
    [
        (np.zeros((10,)), (10,)),
        (np.zeros((5, 5)), (5, 5)),
        (np.zeros(()), ()),
    ],
)
def test_shape(array, result):
    shape = NumpyCompBackend.shape(array)
    assert shape == result
    assert type(shape) == tuple


def test_empty():
    array = NumpyCompBackend.empty((10, 3))
    assert array.shape == (10, 3)
