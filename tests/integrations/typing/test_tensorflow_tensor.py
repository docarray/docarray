import pytest

from docarray import BaseDoc
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore

    from docarray.typing import TensorFlowEmbedding, TensorFlowTensor


@pytest.mark.skipif(not tf_available, reason="Tensorflow not found")
@pytest.mark.tensorflow
def test_set_tensorflow_tensor():
    class MyDocument(BaseDoc):
        t: TensorFlowTensor

    doc = MyDocument(t=tf.zeros((3, 224, 224)))

    assert isinstance(doc.t, TensorFlowTensor)
    assert isinstance(doc.t.tensor, tf.Tensor)
    assert tnp.allclose(doc.t.tensor, tf.zeros((3, 224, 224)))


@pytest.mark.skipif(not tf_available, reason="Tensorflow not found")
@pytest.mark.tensorflow
def test_set_tf_embedding():
    class MyDocument(BaseDoc):
        embedding: TensorFlowEmbedding

    doc = MyDocument(embedding=tf.zeros((128,)))

    assert isinstance(doc.embedding, TensorFlowTensor)
    assert isinstance(doc.embedding, TensorFlowEmbedding)
    assert isinstance(doc.embedding.tensor, tf.Tensor)
    assert tnp.allclose(doc.embedding.tensor, tf.zeros((128,)))
