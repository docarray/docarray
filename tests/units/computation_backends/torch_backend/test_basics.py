import torch

from docarray.computation.torch_backend import TorchCompBackend


def test_to_device():
    t = torch.rand(10, 3)
    assert t.device == torch.device('cpu')
    t = TorchCompBackend.to_device(t, 'meta')
    assert t.device == torch.device('meta')


def test_empty():
    tensor = TorchCompBackend.empty((10, 3))
    assert tensor.shape == (10, 3)
