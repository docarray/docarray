import numpy as np
import torch

from docarray.utility.helper.numpy import top_k as np_top_k
from docarray.utility.helper.torch import top_k as torch_top_k


def test_topk_torch():
    a = torch.tensor([1, 4, 2, 7, 4, 9, 2])
    vals, indices = torch_top_k(a, 3)
    assert vals.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (vals.squeeze() == torch.tensor([1, 2, 2])).all()
    assert (indices.squeeze() == torch.tensor([0, 2, 6])).all() or (
        indices.squeeze() == torch.tensor([0, 6, 2])
    ).all()

    a = torch.tensor([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]])
    vals, indices = torch_top_k(a, 3)
    assert vals.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert (vals[0] == torch.tensor([1, 2, 2])).all()
    assert (indices[0] == torch.tensor([0, 2, 6])).all() or (
        indices[0] == torch.tensor([0, 6, 2])
    ).all()
    assert (vals[1] == torch.tensor([2, 3, 4])).all()
    assert (indices[1] == torch.tensor([2, 4, 6])).all()


def test_topk_numpy():
    a = np.array([1, 4, 2, 7, 4, 9, 2])
    vals, indices = np_top_k(a, 3)
    assert vals.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (vals.squeeze() == np.array([1, 2, 2])).all()
    assert (indices.squeeze() == np.array([0, 2, 6])).all() or (
        indices.squeeze() == np.array([0, 6, 2])
    ).all()

    a = np.array([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]])
    vals, indices = np_top_k(a, 3)
    assert vals.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert (vals[0] == np.array([1, 2, 2])).all()
    assert (indices[0] == np.array([0, 2, 6])).all() or (
        indices[0] == np.array([0, 6, 2])
    ).all()
    assert (vals[1] == np.array([2, 3, 4])).all()
    assert (indices[1] == np.array([2, 4, 6])).all()
