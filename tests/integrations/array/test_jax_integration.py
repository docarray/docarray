from typing import Optional

import jax.numpy as jnp
import pytest
from jax import jit

from docarray import BaseDoc, DocList
from docarray.typing import JaxArray


class Mmdoc(BaseDoc):
    tensor: Optional[JaxArray[3, 224, 224]]


def basic_jax_fn(x):
    return jnp.sum(x)


def abstract_JaxArray(array: JaxArray) -> jnp.ndarray:
    return array.tensor


@pytest.mark.jax
def test_basic_jax_operation():
    N = 10

    batch = DocList[Mmdoc](Mmdoc() for _ in range(N))
    batch.tensor = jnp.zeros((N, 3, 224, 224))

    batch = batch.to_doc_vec()

    jax_fn = jit(basic_jax_fn)
    result = jax_fn(abstract_JaxArray(batch.tensor))

    assert (
        result == 0.0
    )  # checking if the sum of the tensor data is zero as initialized
