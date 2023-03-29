import numpy as np
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


@pytest.mark.parametrize('dtype', [torch.int64, torch.float64, torch.int, torch.float])
def test_dtype(dtype):
    tensor = torch.tensor([1, 2, 3], dtype=dtype)
    assert TorchCompBackend.dtype(tensor) == dtype


def test_device():
    tensor = torch.tensor([1, 2, 3])
    assert TorchCompBackend.device(tensor) == 'cpu'


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


def test_squeeze():
    tensor = torch.zeros(size=(1, 1, 3, 1))
    squeezed = TorchCompBackend.squeeze(tensor)
    assert squeezed.shape == (3,)


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


def test_reshape():
    a = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    b = TorchCompBackend.reshape(a, (2, 3))
    assert torch.equal(b, torch.tensor([[1, 2, 3], [4, 5, 6]]))


def test_copy():
    a = torch.tensor([1, 2, 3])
    b = TorchCompBackend.copy(a)
    assert torch.equal(a, b)


def test_stack():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    stacked = TorchCompBackend.stack([a, b], dim=0)
    assert torch.equal(stacked, torch.tensor([[1, 2, 3], [4, 5, 6]]))


def test_empty_all():
    shape = (2, 3)
    dtype = torch.float32
    device = 'cpu'
    a = TorchCompBackend.empty(shape, dtype, device)
    assert a.shape == shape and a.dtype == dtype and a.device.type == device


def test_to_numpy():
    a = torch.tensor([1, 2, 3])
    b = TorchCompBackend.to_numpy(a)
    assert np.array_equal(b, np.array(a))


def test_none_value():
    assert torch.isnan(TorchCompBackend.none_value())


def test_detach():
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = TorchCompBackend.detach(a)
    assert not b.requires_grad
