// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import pytest

from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.computation.tensorflow_backend import TensorFlowCompBackend
    from docarray.typing import TensorFlowTensor

    metrics = TensorFlowCompBackend.Metrics
else:
    metrics = None


@pytest.mark.tensorflow
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


@pytest.mark.tensorflow
def test_euclidean_dist_tf():
    a = TensorFlowTensor(tf.random.normal((128,)))
    b = TensorFlowTensor(tf.random.normal((128,)))
    assert metrics.euclidean_dist(a, b).tensor.shape == (1,)
    assert metrics.euclidean_dist(a, b).tensor == metrics.euclidean_dist(b, a).tensor
    tf.experimental.numpy.allclose(metrics.euclidean_dist(a, a).tensor, tf.zeros(1))

    a = TensorFlowTensor(tf.zeros((1, 1)))
    b = TensorFlowTensor(tf.ones((4, 1)))
    assert metrics.euclidean_dist(a, b).tensor.shape == (4,)
    tf.experimental.numpy.allclose(
        metrics.euclidean_dist(a, b).tensor, metrics.euclidean_dist(b, a).tensor
    )
    tf.experimental.numpy.allclose(metrics.euclidean_dist(a, a).tensor, tf.zeros(1))

    a = TensorFlowTensor(tf.constant([0.0, 2.0, 0.0]))
    b = TensorFlowTensor(tf.constant([0.0, 0.0, 2.0]))
    desired_output_singleton: tf.Tensor = tf.math.sqrt(
        tf.constant([2.0**2.0 + 2.0**2.0])
    )
    tf.experimental.numpy.allclose(
        metrics.euclidean_dist(a, b).tensor, desired_output_singleton
    )

    a = TensorFlowTensor(tf.constant([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))
    b = TensorFlowTensor(tf.constant([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]))
    desired_output_singleton = tf.constant([[2.828427, 0.0], [0.0, 2.828427]])
    tf.experimental.numpy.allclose(
        metrics.euclidean_dist(a, b).tensor, desired_output_singleton
    )


@pytest.mark.tensorflow
def test_sqeuclidean_dist_torch():
    a = TensorFlowTensor(tf.random.normal((128,)))
    b = TensorFlowTensor(tf.random.normal((128,)))
    assert metrics.sqeuclidean_dist(a, b).tensor.shape == (1,)
    tf.experimental.numpy.allclose(
        metrics.sqeuclidean_dist(a, b).tensor,
        metrics.euclidean_dist(a, b).tensor ** 2,
    )

    a = TensorFlowTensor(tf.random.normal((1, 1)))
    b = TensorFlowTensor(tf.random.normal((4, 1)))
    assert metrics.sqeuclidean_dist(b, a).tensor.shape == (4,)
    tf.experimental.numpy.allclose(
        metrics.sqeuclidean_dist(a, b).tensor,
        metrics.euclidean_dist(a, b).tensor ** 2,
    )
