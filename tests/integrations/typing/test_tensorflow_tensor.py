import pytest

from docarray import BaseDocument

try:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore

    from docarray.typing import TensorFlowTensor
except (ImportError, TypeError):
    pass


@pytest.mark.tensorflow
def test_set_tensorflow_tensor():
    class MyDocument(BaseDocument):
        t: TensorFlowTensor

    doc = MyDocument(t=tf.zeros((3, 224, 224)))

    assert isinstance(doc.t, TensorFlowTensor)
    assert isinstance(doc.t.tensor, tf.Tensor)
    assert tnp.allclose(doc.t.tensor, tf.zeros((3, 224, 224)))
