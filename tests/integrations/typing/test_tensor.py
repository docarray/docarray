import numpy as np
import pytest
import torch

from docarray import BaseDocument
from docarray.typing import AnyTensor, NdArray, TorchTensor

try:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore

    from docarray.typing import TensorFlowTensor
except (ImportError, TypeError):
    TensorFlowTensor = None


def test_set_tensor():
    class MyDocument(BaseDocument):
        tensor: AnyTensor

    d = MyDocument(tensor=np.zeros((3, 224, 224)))

    assert isinstance(d.tensor, NdArray)
    assert isinstance(d.tensor, np.ndarray)
    assert (d.tensor == np.zeros((3, 224, 224))).all()

    d = MyDocument(tensor=torch.zeros((3, 224, 224)))

    assert isinstance(d.tensor, TorchTensor)
    assert isinstance(d.tensor, torch.Tensor)
    assert (d.tensor == torch.zeros((3, 224, 224))).all()


@pytest.mark.tensorflow
def test_set_tensor():
    class MyDocument(BaseDocument):
        tensor: AnyTensor

    d = MyDocument(tensor=tf.zeros((3, 224, 224)))

    assert isinstance(d.tensor, TensorFlowTensor)
    assert isinstance(d.tensor.tensor, tf.Tensor)
    assert tnp.allclose(d.tensor.tensor, tf.zeros((3, 224, 224)))
