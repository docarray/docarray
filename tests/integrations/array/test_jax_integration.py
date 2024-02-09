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
from typing import Optional

import pytest

from docarray import BaseDoc, DocList
from docarray.utils._internal.misc import is_jax_available

if is_jax_available():
    import jax.numpy as jnp
    from jax import jit

    from docarray.typing import JaxArray


@pytest.mark.jax
def test_basic_jax_operation():
    def basic_jax_fn(x):
        return jnp.sum(x)

    def abstract_JaxArray(array: 'JaxArray') -> jnp.ndarray:
        return array.tensor

    class Mmdoc(BaseDoc):
        tensor: Optional[JaxArray[3, 224, 224]] = None

    N = 10

    batch = DocList[Mmdoc](Mmdoc() for _ in range(N))
    batch.tensor = jnp.zeros((N, 3, 224, 224))

    batch = batch.to_doc_vec()

    jax_fn = jit(basic_jax_fn)
    result = jax_fn(abstract_JaxArray(batch.tensor))

    assert (
        result == 0.0
    )  # checking if the sum of the tensor data is zero as initialized
