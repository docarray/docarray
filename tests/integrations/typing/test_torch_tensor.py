import torch

from docarray import Document
from docarray.typing import TorchEmbedding, TorchTensor


def test_set_torch_tensor():
    class MyDocument(Document):
        tensor: TorchTensor
        embedding: TorchEmbedding

    d = MyDocument(tensor=torch.zeros((3, 224, 224)), embedding=torch.zeros((128,)))

    assert isinstance(d.tensor, TorchTensor)
    assert isinstance(d.tensor, torch.Tensor)
    assert (d.tensor == torch.zeros((3, 224, 224))).all()

    assert isinstance(d.embedding, TorchTensor)
    assert isinstance(d.embedding, TorchEmbedding)
    assert isinstance(d.embedding, torch.Tensor)
    assert (d.embedding == torch.zeros((128,))).all()
