import torch
from docarray.typing.tensor.torch_tensor import TorchTensor
import copy

from docarray import BaseDoc
from docarray.typing import TorchEmbedding, TorchTensor


def test_set_torch_tensor():
    class MyDocument(BaseDoc):
        tensor: TorchTensor

    d = MyDocument(tensor=torch.zeros((3, 224, 224)))

    assert isinstance(d.tensor, TorchTensor)
    assert isinstance(d.tensor, torch.Tensor)
    assert (d.tensor == torch.zeros((3, 224, 224))).all()


def test_set_torch_embedding():
    class MyDocument(BaseDoc):
        embedding: TorchEmbedding

    d = MyDocument(embedding=torch.zeros((128,)))

    assert isinstance(d.embedding, TorchTensor)
    assert isinstance(d.embedding, TorchEmbedding)
    assert isinstance(d.embedding, torch.Tensor)
    assert (d.embedding == torch.zeros((128,))).all()

def test_torchtensor_deepcopy():
    # Setup
    original_tensor_float = TorchTensor(torch.rand(10))
    original_tensor_int = TorchTensor(torch.randint(0, 100, (10,)))

    # Exercise
    copied_tensor_float = copy.deepcopy(original_tensor_float)
    copied_tensor_int = copy.deepcopy(original_tensor_int)

    # Verify
    assert torch.equal(original_tensor_float, copied_tensor_float)
    assert original_tensor_float is not copied_tensor_float
    assert torch.equal(original_tensor_int, copied_tensor_int)
    assert original_tensor_int is not copied_tensor_int