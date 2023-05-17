import numpy as np
import pytest
import torch

from docarray import BaseDoc
from docarray.computation.tensorflow_backend import tnp
from docarray.typing import AnyTensor, NdArray, TensorFlowTensor, TorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf


@pytest.mark.parametrize(
    'tensor,cls_audio_tensor,cls_tensor',
    [
        (torch.zeros(1000, 2), TorchTensor, torch.Tensor),
        (np.zeros((1000, 2)), NdArray, np.ndarray),
    ],
)
def test_torch_ndarray_to_any_tensor(tensor, cls_audio_tensor, cls_tensor):
    class MyAudioDoc(BaseDoc):
        tensor: AnyTensor

    doc = MyAudioDoc(tensor=tensor)
    assert isinstance(doc.tensor, cls_audio_tensor)
    assert isinstance(doc.tensor, cls_tensor)
    assert (doc.tensor == tensor).all()


@pytest.mark.tensorflow
def test_tensorflow_to_any_tensor():
    class MyAudioDoc(BaseDoc):
        tensor: AnyTensor

    doc = MyAudioDoc(tensor=tf.zeros((1000, 2)))
    assert isinstance(doc.tensor, TensorFlowTensor)
    assert isinstance(doc.tensor.tensor, tf.Tensor)
    assert tnp.allclose(doc.tensor.tensor, tf.zeros((1000, 2)))
