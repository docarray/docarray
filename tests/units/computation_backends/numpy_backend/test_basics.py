import numpy as np
import pytest
from pydantic import parse_obj_as

from docarray.computation.numpy_backend import NumpyCompBackend
from docarray.typing import NdArray


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


def test_device():
    array = np.array([1, 2, 3])
    assert NumpyCompBackend.device(array) is None


@pytest.mark.parametrize('dtype', [np.int64, np.float64, np.int, np.float])
def test_dtype(dtype):
    array = np.array([1, 2, 3], dtype=dtype)
    assert NumpyCompBackend.dtype(array) == dtype


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


def test_squeeze():
    tensor = np.zeros(shape=(1, 1, 3, 1))
    squeezed = NumpyCompBackend.squeeze(tensor)
    assert squeezed.shape == (3,)


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


def test_stack():
    t0 = parse_obj_as(NdArray, np.zeros((3, 224, 224)))
    t1 = parse_obj_as(NdArray, np.ones((3, 224, 224)))

    stacked1 = NumpyCompBackend.stack([t0, t1], dim=0)
    assert isinstance(stacked1, np.ndarray)
    assert stacked1.shape == (2, 3, 224, 224)

    stacked2 = NumpyCompBackend.stack([t0, t1], dim=-1)
    assert isinstance(stacked2, np.ndarray)
    assert stacked2.shape == (3, 224, 224, 2)
