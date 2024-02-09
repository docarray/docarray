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
    print("is jax available", jax_available)
    import jax
    import jax.numpy as jnp

    from docarray.computation.jax_backend import JaxCompBackend
    from docarray.typing import JaxArray

    jax.config.update("jax_enable_x64", True)


@pytest.mark.jax
@pytest.mark.parametrize(
    'shape,result',
    [
        ((5), 1),
        ((1, 5), 2),
        ((5, 5), 2),
        ((), 0),
    ],
)
def test_n_dim(shape, result):

    array = JaxArray(jnp.zeros(shape))
    assert JaxCompBackend.n_dim(array) == result


@pytest.mark.jax
@pytest.mark.parametrize(
    'shape,result',
    [
        ((10,), (10,)),
        ((5, 5), (5, 5)),
        ((), ()),
    ],
)
def test_shape(shape, result):
    array = JaxArray(jnp.zeros(shape))
    shape = JaxCompBackend.shape(array)
    assert shape == result
    assert type(shape) == tuple


@pytest.mark.jax
def test_to_device():
    array = JaxArray(jnp.zeros((3)))
    array = JaxCompBackend.to_device(array, 'cpu')
    assert array.tensor.device().platform.endswith('cpu')


@pytest.mark.jax
@pytest.mark.parametrize(
    'dtype,result_type',
    [
        ('int64', 'int64'),
        ('float64', 'float64'),
        ('int8', 'int8'),
        ('double', 'float64'),
    ],
)
def test_dtype(dtype, result_type):
    array = JaxArray(jnp.array([1, 2, 3], dtype=dtype))
    assert JaxCompBackend.dtype(array) == result_type


@pytest.mark.jax
def test_empty():
    array = JaxCompBackend.empty((10, 3))
    assert array.tensor.shape == (10, 3)


@pytest.mark.jax
def test_empty_dtype():
    tf_tensor = JaxCompBackend.empty((10, 3), dtype=jnp.int32)
    assert tf_tensor.tensor.shape == (10, 3)
    assert tf_tensor.tensor.dtype == jnp.int32


@pytest.mark.jax
def test_empty_device():
    tensor = JaxCompBackend.empty((10, 3), device='cpu')
    assert tensor.tensor.shape == (10, 3)
    assert tensor.tensor.device().platform.endswith('cpu')


@pytest.mark.jax
def test_squeeze():
    tensor = JaxArray(jnp.zeros(shape=(1, 1, 3, 1)))
    squeezed = JaxCompBackend.squeeze(tensor)
    assert squeezed.tensor.shape == (3,)


@pytest.mark.jax
@pytest.mark.parametrize(
    'data_input,t_range,x_range,data_result',
    [
        (
            [0, 1, 2, 3, 4, 5],
            (0, 10),
            None,
            [0, 2, 4, 6, 8, 10],
        ),
        (
            [0, 1, 2, 3, 4, 5],
            (0, 10),
            (0, 10),
            [0, 1, 2, 3, 4, 5],
        ),
        (
            [[0.0, 1.0], [0.0, 1.0]],
            (0, 10),
            None,
            [[0.0, 10.0], [0.0, 10.0]],
        ),
    ],
)
def test_minmax_normalize(data_input, t_range, x_range, data_result):
    array = JaxArray(jnp.array(data_input))
    output = JaxCompBackend.minmax_normalize(
        tensor=array, t_range=t_range, x_range=x_range
    )
    assert jnp.allclose(output.tensor, jnp.array(data_result))


@pytest.mark.jax
def test_reshape():
    tensor = JaxArray(jnp.zeros((3, 224, 224)))
    reshaped = JaxCompBackend.reshape(tensor, (224, 224, 3))
    assert reshaped.tensor.shape == (224, 224, 3)


@pytest.mark.jax
def test_stack():
    t0 = JaxArray(jnp.zeros((3, 224, 224)))
    t1 = JaxArray(jnp.ones((3, 224, 224)))

    stacked1 = JaxCompBackend.stack([t0, t1], dim=0)
    assert isinstance(stacked1, JaxArray)
    assert stacked1.tensor.shape == (2, 3, 224, 224)

    stacked2 = JaxCompBackend.stack([t0, t1], dim=-1)
    assert isinstance(stacked2, JaxArray)
    assert stacked2.tensor.shape == (3, 224, 224, 2)
