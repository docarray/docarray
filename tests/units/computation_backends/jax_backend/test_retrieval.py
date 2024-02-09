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
    import jax.numpy as jnp

    from docarray.computation.jax_backend import JaxCompBackend
    from docarray.typing import JaxArray

    metrics = JaxCompBackend.Metrics
else:
    metrics = None


@pytest.mark.jax
def test_top_k_descending_false():
    top_k = JaxCompBackend.Retrieval.top_k

    a = JaxArray(jnp.array([1, 4, 2, 7, 4, 9, 2]))
    vals, indices = top_k(a, 3, descending=False)

    assert vals.tensor.shape == (1, 3)
    assert indices.tensor.shape == (1, 3)
    assert jnp.allclose(jnp.squeeze(vals.tensor), jnp.array([1, 2, 2]))
    assert jnp.allclose(jnp.squeeze(indices.tensor), jnp.array([0, 2, 6])) or (
        jnp.allclose(jnp.squeeze.indices.tensor),
        jnp.array([0, 6, 2]),
    )

    a = JaxArray(jnp.array([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]]))
    vals, indices = top_k(a, 3, descending=False)
    assert vals.tensor.shape == (2, 3)
    assert indices.tensor.shape == (2, 3)
    assert jnp.allclose(vals.tensor[0], jnp.array([1, 2, 2]))
    assert jnp.allclose(indices.tensor[0], jnp.array([0, 2, 6])) or jnp.allclose(
        indices.tensor[0], jnp.array([0, 6, 2])
    )
    assert jnp.allclose(vals.tensor[1], jnp.array([2, 3, 4]))
    assert jnp.allclose(indices.tensor[1], jnp.array([2, 4, 6]))


@pytest.mark.jax
def test_top_k_descending_true():
    top_k = JaxCompBackend.Retrieval.top_k

    a = JaxArray(jnp.array([1, 4, 2, 7, 4, 9, 2]))
    vals, indices = top_k(a, 3, descending=True)

    assert vals.tensor.shape == (1, 3)
    assert indices.tensor.shape == (1, 3)
    assert jnp.allclose(jnp.squeeze(vals.tensor), jnp.array([9, 7, 4]))
    assert jnp.allclose(jnp.squeeze(indices.tensor), jnp.array([5, 3, 1]))

    a = JaxArray(jnp.array([[1, 4, 2, 7, 4, 9, 2], [11, 6, 2, 7, 3, 10, 4]]))
    vals, indices = top_k(a, 3, descending=True)

    assert vals.tensor.shape == (2, 3)
    assert indices.tensor.shape == (2, 3)

    assert jnp.allclose(vals.tensor[0], jnp.array([9, 7, 4]))
    assert jnp.allclose(indices.tensor[0], jnp.array([5, 3, 1]))

    assert jnp.allclose(vals.tensor[1], jnp.array([11, 10, 7]))
    assert jnp.allclose(indices.tensor[1], jnp.array([0, 5, 3]))
