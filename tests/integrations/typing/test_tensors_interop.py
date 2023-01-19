import numpy as np
import torch

from docarray.typing import (
    AudioNdArray,
    AudioTorchTensor,
    NdArrayEmbedding,
    TorchEmbedding,
)


def test_torch_tensors_interop():
    t1 = AudioTorchTensor(torch.rand(128))
    t2 = TorchEmbedding(torch.rand(128))

    t_result = t1 + t2
    assert isinstance(t_result, AudioTorchTensor)
    assert isinstance(t_result, torch.Tensor)
    assert t_result.shape == (128,)


def test_np_arrays_interop():
    t1 = AudioNdArray((128,))
    t2 = NdArrayEmbedding((128,))

    t_result = t1 + t2
    assert isinstance(t_result, AudioNdArray)
    assert isinstance(t_result, np.ndarray)
    assert t_result.shape == (128,)
