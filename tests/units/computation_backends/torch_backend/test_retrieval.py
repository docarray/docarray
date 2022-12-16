import torch

from docarray.computation.torch_backend import TorchCompBackend


def test_topk():
    top_k = TorchCompBackend.Retrieval.top_k

    a = torch.tensor([1, 4, 2, 7, 4, 9, 2])
    vals, indices = top_k(a, 3)
    assert vals.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (vals.squeeze() == torch.tensor([1, 2, 2])).all()
    assert (indices.squeeze() == torch.tensor([0, 2, 6])).all() or (
        indices.squeeze() == torch.tensor([0, 6, 2])
    ).all()

    a = torch.tensor([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]])
    vals, indices = top_k(a, 3)
    assert vals.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert (vals[0] == torch.tensor([1, 2, 2])).all()
    assert (indices[0] == torch.tensor([0, 2, 6])).all() or (
        indices[0] == torch.tensor([0, 6, 2])
    ).all()
    assert (vals[1] == torch.tensor([2, 3, 4])).all()
    assert (indices[1] == torch.tensor([2, 4, 6])).all()
