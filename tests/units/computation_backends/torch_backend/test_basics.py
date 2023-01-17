import pytest
import torch

from docarray.computation.torch_backend import TorchCompBackend


def test_to_device():
    t = torch.rand(10, 3)
    assert t.device == torch.device('cpu')
    t = TorchCompBackend.to_device(t, 'meta')
    assert t.device == torch.device('meta')


@pytest.mark.parametrize(
    'array,result',
    [
        (torch.zeros((5)), 1),
        (torch.zeros((1, 5)), 2),
        (torch.zeros((5, 5)), 2),
        (torch.zeros(()), 0),
    ],
)
def test_n_dim(array, result):
    assert TorchCompBackend.n_dim(array) == result


@pytest.mark.parametrize(
    'array,result',
    [
        (torch.zeros((10,)), (10,)),
        (torch.zeros((5, 5)), (5, 5)),
        (torch.zeros(()), ()),
    ],
)
def test_shape(array, result):
    shape = TorchCompBackend.shape(array)
    assert shape == result
    assert type(shape) == tuple


def test_empty():
    tensor = TorchCompBackend.empty((10, 3))
    assert tensor.shape == (10, 3)
