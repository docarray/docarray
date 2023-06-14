import jax.numpy as jnp
import numpy as np
import pytest

from docarray.computation.jax_backend import JaxCompBackend
from docarray.typing import JaxArray


@pytest.mark.tensorflow
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


@pytest.mark.tensorflow
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


@pytest.mark.tensorflow
def test_to_device():
    array = JaxArray(jnp.constant([1, 2, 3]))
    array = JaxCompBackend.to_device(array, 'CPU:0')
    assert array.tensor.device.endswith('CPU:0')


@pytest.mark.tensorflow
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
    array = JaxArray(jnp.constant([1, 2, 3], dtype=getattr(jnp, dtype)))
    assert JaxCompBackend.dtype(array) == result_type


@pytest.mark.tensorflow
def test_empty():
    array = JaxCompBackend.empty((10, 3))
    assert array.tensor.shape == (10, 3)


@pytest.mark.tensorflow
def test_empty_dtype():
    tf_tensor = JaxCompBackend.empty((10, 3), dtype=jnp.int32)
    assert tf_tensor.tensor.shape == (10, 3)
    assert tf_tensor.tensor.dtype == jnp.int32


@pytest.mark.tensorflow
def test_empty_device():
    tensor = JaxCompBackend.empty((10, 3), device='CPU:0')
    assert tensor.tensor.shape == (10, 3)
    assert tensor.tensor.device.endswith('CPU:0')


@pytest.mark.tensorflow
def test_squeeze():
    tensor = JaxArray(jnp.zeros(shape=(1, 1, 3, 1)))
    squeezed = JaxCompBackend.squeeze(tensor)
    assert squeezed.tensor.shape == (3,)


@pytest.mark.tensorflow
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
    array = JaxArray(jnp.constant(data_input))
    output = JaxCompBackend.minmax_normalize(
        tensor=array, t_range=t_range, x_range=x_range
    )
    assert np.allclose(output.tensor, jnp.constant(data_result))


@pytest.mark.tensorflow
def test_reshape():
    tensor = JaxArray(jnp.zeros((3, 224, 224)))
    reshaped = JaxCompBackend.reshape(tensor, (224, 224, 3))
    assert reshaped.tensor.shape == (224, 224, 3)


@pytest.mark.tensorflow
def test_stack():
    t0 = JaxArray(jnp.zeros((3, 224, 224)))
    t1 = JaxArray(jnp.ones((3, 224, 224)))

    stacked1 = JaxCompBackend.stack([t0, t1], dim=0)
    assert isinstance(stacked1, JaxArray)
    assert stacked1.tensor.shape == (2, 3, 224, 224)

    stacked2 = JaxCompBackend.stack([t0, t1], dim=-1)
    assert isinstance(stacked2, JaxArray)
    assert stacked2.tensor.shape == (3, 224, 224, 2)
