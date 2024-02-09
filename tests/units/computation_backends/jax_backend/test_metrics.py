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

from docarray.utils._internal.misc import is_jax_available

jax_available = is_jax_available()
if jax_available:
    import jax
    import jax.numpy as jnp

    from docarray.computation.jax_backend import JaxCompBackend
    from docarray.typing import JaxArray

    metrics = JaxCompBackend.Metrics
else:
    metrics = None


@pytest.mark.jax
def test_cosine_sim_jax():
    a = JaxArray(jax.random.uniform(jax.random.PRNGKey(0), shape=(128,)))
    b = JaxArray(jax.random.uniform(jax.random.PRNGKey(1), shape=(128,)))
    assert metrics.cosine_sim(a, b).tensor.shape == (1,)
    assert metrics.cosine_sim(a, b).tensor == metrics.cosine_sim(b, a).tensor

    assert jnp.allclose(metrics.cosine_sim(a, a).tensor, jnp.ones((1,)))

    a = JaxArray(jax.random.uniform(jax.random.PRNGKey(2), shape=(10, 3)))
    b = JaxArray(jax.random.uniform(jax.random.PRNGKey(3), shape=(5, 3)))
    assert metrics.cosine_sim(a, b).tensor.shape == (10, 5)
    assert metrics.cosine_sim(b, a).tensor.shape == (5, 10)
    diag_dists = jnp.diagonal(metrics.cosine_sim(b, b).tensor)  # self-comparisons
    assert jnp.allclose(diag_dists, jnp.ones((5,)))


@pytest.mark.jax
@pytest.mark.skip
def test_euclidean_dist_jax():
    a = JaxArray(jax.random.normal(jax.random.PRNGKey(0), shape=(128,)))
    b = JaxArray(jax.random.normal(jax.random.PRNGKey(1), shape=(128,)))
    assert metrics.euclidean_dist(a, b).tensor.shape == (1,)
    assert jnp.allclose(
        metrics.euclidean_dist(a, b).tensor, metrics.euclidean_dist(b, a).tensor
    )

    assert jnp.allclose(metrics.euclidean_dist(a, a).tensor, jnp.zeros((1,)))

    a = JaxArray(jnp.zeros((1, 1)))
    b = JaxArray(jnp.ones((4, 1)))
    assert metrics.euclidean_dist(a, b).tensor.shape == (4,)
    assert jnp.allclose(
        metrics.euclidean_dist(a, b).tensor, metrics.euclidean_dist(b, a).tensor
    )
    assert jnp.allclose(metrics.euclidean_dist(a, a).tensor, jnp.zeros((1,)))

    a = JaxArray(jnp.array([0.0, 2.0, 0.0]))
    b = JaxArray(jnp.array([0.0, 0.0, 2.0]))
    desired_output_singleton = jnp.sqrt(jnp.array([2.0**2.0 + 2.0**2.0]))
    assert jnp.allclose(metrics.euclidean_dist(a, b).tensor, desired_output_singleton)

    a = JaxArray(jnp.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))
    b = JaxArray(jnp.array([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]))
    desired_output_singleton = jnp.array([[2.828427, 0.0], [0.0, 2.828427]])

    assert jnp.allclose(metrics.euclidean_dist(a, b).tensor, desired_output_singleton)


@pytest.mark.jax
def test_sqeuclidea_dist_jnp():
    a = JaxArray(jax.random.uniform(jax.random.PRNGKey(0), shape=(128,)))
    b = JaxArray(jax.random.uniform(jax.random.PRNGKey(1), shape=(128,)))
    assert metrics.sqeuclidean_dist(a, b).tensor.shape == (1,)
    assert jnp.allclose(
        metrics.sqeuclidean_dist(a, b).tensor, metrics.euclidean_dist(a, b).tensor ** 2
    )

    a = JaxArray(jax.random.uniform(jax.random.PRNGKey(2), shape=(10, 3)))
    b = JaxArray(jax.random.uniform(jax.random.PRNGKey(3), shape=(5, 3)))
    assert metrics.sqeuclidean_dist(a, b).tensor.shape == (10, 5)
    assert jnp.allclose(
        metrics.sqeuclidean_dist(a, b).tensor, metrics.euclidean_dist(a, b).tensor ** 2
    )
