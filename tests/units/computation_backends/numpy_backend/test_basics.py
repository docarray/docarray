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


def test_empty_dtype():
    tensor = NumpyCompBackend.empty((10, 3), dtype=np.int32)
    assert tensor.shape == (10, 3)
    assert tensor.dtype == np.int32


def test_empty_device():
    with pytest.raises(NotImplementedError):
        NumpyCompBackend.empty((10, 3), device='meta')


@pytest.mark.parametrize(
    'array,t_range,x_range,result',
    [
        (np.array([0, 1, 2, 3, 4, 5]), (0, 10), None, np.array([0, 2, 4, 6, 8, 10])),
        (np.array([0, 1, 2, 3, 4, 5]), (0, 10), (0, 10), np.array([0, 1, 2, 3, 4, 5])),
        (
            np.array([[0.0, 1.0], [0.0, 1.0]]),
            (0, 10),
            None,
            np.array([[0.0, 10.0], [0.0, 10.0]]),
        ),
    ],
)
def test_minmax_normalize(array, t_range, x_range, result):
    output = NumpyCompBackend.minmax_normalize(
        tensor=array, t_range=t_range, x_range=x_range
    )
    assert np.allclose(output, result)
