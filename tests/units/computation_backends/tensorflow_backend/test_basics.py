import numpy as np
import pytest

from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf

    from docarray.computation.tensorflow_backend import TensorFlowCompBackend
    from docarray.typing import TensorFlowTensor


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
    array = TensorFlowTensor(tf.zeros(shape))
    assert TensorFlowCompBackend.n_dim(array) == result


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
    array = TensorFlowTensor(tf.zeros(shape))
    shape = TensorFlowCompBackend.shape(array)
    assert shape == result
    assert type(shape) == tuple


@pytest.mark.tensorflow
def test_to_device():
    array = TensorFlowTensor(tf.constant([1, 2, 3]))
    array = TensorFlowCompBackend.to_device(array, 'CPU:0')
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
    array = TensorFlowTensor(tf.constant([1, 2, 3], dtype=getattr(tf, dtype)))
    assert TensorFlowCompBackend.dtype(array) == result_type


@pytest.mark.tensorflow
def test_empty():
    array = TensorFlowCompBackend.empty((10, 3))
    assert array.tensor.shape == (10, 3)


@pytest.mark.tensorflow
def test_empty_dtype():
    tf_tensor = TensorFlowCompBackend.empty((10, 3), dtype=tf.int32)
    assert tf_tensor.tensor.shape == (10, 3)
    assert tf_tensor.tensor.dtype == tf.int32


@pytest.mark.tensorflow
def test_empty_device():
    tensor = TensorFlowCompBackend.empty((10, 3), device='CPU:0')
    assert tensor.tensor.shape == (10, 3)
    assert tensor.tensor.device.endswith('CPU:0')


@pytest.mark.tensorflow
def test_squeeze():
    tensor = TensorFlowTensor(tf.zeros(shape=(1, 1, 3, 1)))
    squeezed = TensorFlowCompBackend.squeeze(tensor)
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
    array = TensorFlowTensor(tf.constant(data_input))
    output = TensorFlowCompBackend.minmax_normalize(
        tensor=array, t_range=t_range, x_range=x_range
    )
    assert np.allclose(output.tensor, tf.constant(data_result))


@pytest.mark.tensorflow
def test_reshape():
    tensor = TensorFlowTensor(tf.zeros((3, 224, 224)))
    reshaped = TensorFlowCompBackend.reshape(tensor, (224, 224, 3))
    assert reshaped.tensor.shape == (224, 224, 3)


@pytest.mark.tensorflow
def test_stack():
    t0 = TensorFlowTensor(tf.zeros((3, 224, 224)))
    t1 = TensorFlowTensor(tf.ones((3, 224, 224)))

    stacked1 = TensorFlowCompBackend.stack([t0, t1], dim=0)
    assert isinstance(stacked1, TensorFlowTensor)
    assert stacked1.tensor.shape == (2, 3, 224, 224)

    stacked2 = TensorFlowCompBackend.stack([t0, t1], dim=-1)
    assert isinstance(stacked2, TensorFlowTensor)
    assert stacked2.tensor.shape == (3, 224, 224, 2)
