import tensorflow as tf

from docarray.computation.tensorflow_backend import TensorFlowCompBackend
from docarray.typing import TensorFlowTensor

metrics = TensorFlowCompBackend.Metrics


def test_cosine_sim_tf():
    a = TensorFlowTensor(tf.random.normal((128,)))
    b = TensorFlowTensor(tf.random.normal((128,)))
    assert metrics.cosine_sim(a, b).tensor.shape == (1,)
    assert metrics.cosine_sim(a, b).tensor == metrics.cosine_sim(b, a).tensor
    tf.experimental.numpy.allclose(metrics.cosine_sim(a, a).tensor, tf.ones(1))

    a = TensorFlowTensor(tf.random.normal((10, 3)))
    b = TensorFlowTensor(tf.random.normal((5, 3)))
    assert metrics.cosine_sim(a, b).tensor.shape == (10, 5)
    assert metrics.cosine_sim(b, a).tensor.shape == (5, 10)
    diag_dists = tf.linalg.diag(metrics.cosine_sim(b, b).tensor)  # self-comparisons
    tf.experimental.numpy.allclose(diag_dists, tf.ones(5))
