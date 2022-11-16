import torch

from docarray import Document
from docarray.typing import TorchTensor


def test_set_tensor():
    class MyDocument(Document):
        tensor: TorchTensor

    d = MyDocument(tensor=torch.zeros((3, 224, 224)))

    assert isinstance(d.tensor, TorchTensor)
    assert isinstance(d.tensor, torch.Tensor)
    assert (d.tensor == torch.zeros((3, 224, 224))).all()
