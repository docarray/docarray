import numpy as np
import pytest

from docarray.computation.numpy_backend import NumpyCompBackend


def test_to_device():
    with pytest.raises(NotImplementedError):
        NumpyCompBackend.to_device(np.random.rand(10, 3), 'meta')


def test_empty():
    array = NumpyCompBackend.empty((10, 3))
    assert array.shape == (10, 3)
