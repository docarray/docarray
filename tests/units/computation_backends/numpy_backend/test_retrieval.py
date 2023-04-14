import numpy as np

from docarray.computation.numpy_backend import NumpyCompBackend


def test_topk_numpy():
    top_k = NumpyCompBackend.Retrieval.top_k

    a = np.array([1, 4, 2, 7, 4, 9, 2])
    vals, indices = top_k(a, 3)
    assert vals.shape == (1, 3)
    assert indices.shape == (1, 3)
    assert (vals.squeeze() == np.array([1, 2, 2])).all()
    assert (indices.squeeze() == np.array([0, 2, 6])).all() or (
        indices.squeeze() == np.array([0, 6, 2])
    ).all()

    a = np.array([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]])
    vals, indices = top_k(a, 3)
    assert vals.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert (vals[0] == np.array([1, 2, 2])).all()
    assert (indices[0] == np.array([0, 2, 6])).all() or (
        indices[0] == np.array([0, 6, 2])
    ).all()
    assert (vals[1] == np.array([2, 3, 4])).all()
    assert (indices[1] == np.array([2, 4, 6])).all()
