import torch

from docarray import BaseDocument
from docarray.typing import TorchTensor


def test_tensor_ops():
    class A(BaseDocument):
        tensor: TorchTensor[3, 224, 224]

    class B(BaseDocument):
        tensor: TorchTensor[3, 112, 224]

    tensor = A(tensor=torch.ones(3, 224, 224)).tensor
    tensord = A(tensor=torch.ones(3, 224, 224)).tensor
    tensorn = torch.zeros(3, 224, 224)
    tensorhalf = B(tensor=torch.ones(3, 112, 224)).tensor
    tensorfull = torch.cat([tensorhalf, tensorhalf], dim=1)

    assert type(tensor) == TorchTensor
    assert type(tensor + tensord) == TorchTensor
    assert type(tensor + tensorn) == TorchTensor
    assert type(tensor + tensorfull) == TorchTensor
