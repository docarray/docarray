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


def test_empty_dtype():
    tensor = TorchCompBackend.empty((10, 3), dtype=torch.int32)
    assert tensor.shape == (10, 3)
    assert tensor.dtype == torch.int32


def test_empty_device():
    tensor = TorchCompBackend.empty((10, 3), device='meta')
    assert tensor.shape == (10, 3)
    assert tensor.device == torch.device('meta')


@pytest.mark.parametrize(
    'array,t_range,x_range,result',
    [
        (
            torch.tensor([0, 1, 2, 3, 4, 5]),
            (0, 10),
            None,
            torch.tensor([0, 2, 4, 6, 8, 10]),
        ),
        (
            torch.tensor([0, 1, 2, 3, 4, 5]),
            (0, 10),
            (0, 10),
            torch.tensor([0, 1, 2, 3, 4, 5]),
        ),
        (
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            (0, 10),
            None,
            torch.tensor([[0.0, 10.0], [0.0, 10.0]]),
        ),
    ],
)
def test_minmax_normalize(array, t_range, x_range, result):
    output = TorchCompBackend.minmax_normalize(
        tensor=array, t_range=t_range, x_range=x_range
    )
    assert torch.allclose(output, result)
