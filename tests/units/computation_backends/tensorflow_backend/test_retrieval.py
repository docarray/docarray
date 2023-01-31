import tensorflow as tf
import tensorflow._api.v2.experimental.numpy as tnp

from docarray.computation.tensorflow_backend import TensorFlowCompBackend
from docarray.typing import TensorFlowTensor


def test_top_k_descending_false():
    top_k = TensorFlowCompBackend.Retrieval.top_k

    a = TensorFlowTensor(tf.constant([1, 4, 2, 7, 4, 9, 2]))
    vals, indices = top_k(a, 3, descending=False)

    assert vals.tensor.shape == (1, 3)
    assert indices.tensor.shape == (1, 3)
    assert tnp.allclose(tnp.squeeze(vals.tensor), tf.constant([1, 2, 2]))
    assert tnp.allclose(tnp.squeeze(indices.tensor), tf.constant([0, 2, 6])) or (
        tnp.allclose(tnp.squeeze.indices.tensor),
        tf.constant([0, 6, 2]),
    )

    a = TensorFlowTensor(tf.constant([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]]))
    vals, indices = top_k(a, 3, descending=False)
    assert vals.tensor.shape == (2, 3)
    assert indices.tensor.shape == (2, 3)
    assert tnp.allclose(vals.tensor[0], tf.constant([1, 2, 2]))
    assert tnp.allclose(indices.tensor[0], tf.constant([0, 2, 6])) or tnp.allclose(
        indices.tensor[0], tf.constant([0, 6, 2])
    )
    assert tnp.allclose(vals.tensor[1], tf.constant([2, 3, 4]))
    assert tnp.allclose(indices.tensor[1], tf.constant([2, 4, 6]))


def test_top_k_descending_true():
    top_k = TensorFlowCompBackend.Retrieval.top_k

    a = TensorFlowTensor(tf.constant([1, 4, 2, 7, 4, 9, 2]))
    vals, indices = top_k(a, 3, descending=True)

    assert vals.tensor.shape == (1, 3)
    assert indices.tensor.shape == (1, 3)
    assert tnp.allclose(tnp.squeeze(vals.tensor), tf.constant([9, 7, 4]))
    assert tnp.allclose(tnp.squeeze(indices.tensor), tf.constant([5, 3, 1]))

    a = TensorFlowTensor(tf.constant([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]]))
    vals, indices = top_k(a, 3, descending=True)

    assert vals.tensor.shape == (2, 3)
    assert indices.tensor.shape == (2, 3)

    assert tnp.allclose(vals.tensor[0], tf.constant([9, 7, 4]))
    assert tnp.allclose(indices.tensor[0], tf.constant([0, 2, 6]))

    assert tnp.allclose(vals.tensor[1], tf.constant([11, 10, 7]))
    assert tnp.allclose(indices.tensor[1], tf.constant([0, 5, 3]))
