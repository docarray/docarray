import jax.numpy as jnp
import pytest

from docarray.computation.jax_backend import JaxCompBackend
from docarray.typing import JaxArray


@pytest.mark.tensorflow
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


@pytest.mark.tensorflow
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
