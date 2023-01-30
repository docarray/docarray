import numpy as np
import pytest
import tensorflow as tf

from docarray.computation.tensorflow_backend import TensorFlowCompBackend


def test_to_device():
    with pytest.raises(NotImplementedError):
        TensorFlowCompBackend.to_device(tf.zeros((10, 3)), 'CPU:0')


@pytest.mark.parametrize(
    'array,result',
    [
        (tf.zeros((5)), 1),
        (tf.zeros((1, 5)), 2),
        (tf.zeros((5, 5)), 2),
        (tf.zeros(()), 0),
    ],
)
def test_n_dim(array, result):
    assert TensorFlowCompBackend.n_dim(array) == result


@pytest.mark.parametrize(
    'array,result',
    [
        (tf.zeros((10,)), (10,)),
        (tf.zeros((5, 5)), (5, 5)),
        (tf.zeros(()), ()),
    ],
)
def test_shape(array, result):
    shape = TensorFlowCompBackend.shape(array)
    assert shape == result
    assert type(shape) == tuple


# def test_device():
#     array = tf.constant([1, 2, 3])
#     assert TensorFlowCompBackend.device(array) is not None


@pytest.mark.parametrize('dtype', [tf.int64, tf.float64, tf.int8, tf.double])
def test_dtype(dtype):
    array = tf.constant([1, 2, 3], dtype=dtype)
    assert TensorFlowCompBackend.dtype(array) == dtype


def test_empty():
    array = TensorFlowCompBackend.empty((10, 3))
    assert array.shape == (10, 3)


def test_empty_dtype():
    tensor = TensorFlowCompBackend.empty((10, 3), dtype=tf.int32)
    assert tensor.shape == (10, 3)
    assert tensor.dtype == tf.int32


# def test_empty_device():
#     with pytest.raises(NotImplementedError):
#         TensorFlowCompBackend.empty((10, 3), device='CPU:0')


def test_squeeze():
    tensor = tf.zeros(shape=(1, 1, 3, 1))
    squeezed = TensorFlowCompBackend.squeeze(tensor)
    assert squeezed.shape == (3,)


@pytest.mark.parametrize(
    'array,t_range,x_range,result',
    [
        (
            tf.constant([0, 1, 2, 3, 4, 5]),
            (0, 10),
            None,
            tf.constant([0, 2, 4, 6, 8, 10]),
        ),
        (
            tf.constant([0, 1, 2, 3, 4, 5]),
            (0, 10),
            (0, 10),
            tf.constant([0, 1, 2, 3, 4, 5]),
        ),
        (
            tf.constant([[0.0, 1.0], [0.0, 1.0]]),
            (0, 10),
            None,
            tf.constant([[0.0, 10.0], [0.0, 10.0]]),
        ),
    ],
)
def test_minmax_normalize(array, t_range, x_range, result):
    output = TensorFlowCompBackend.minmax_normalize(
        tensor=array, t_range=t_range, x_range=x_range
    )
    assert np.allclose(output, result)


def test_reshape():
    tensor = tf.zeros((3, 224, 224))
    reshaped = TensorFlowCompBackend.reshape(tensor, (224, 224, 3))
    assert reshaped.shape == (224, 224, 3)


def test_stack():
    t0 = tf.zeros((3, 224, 224))
    t1 = tf.ones((3, 224, 224))

    stacked1 = TensorFlowCompBackend.stack([t0, t1], dim=0)
    from tensorflow.python.framework.ops import EagerTensor

    assert isinstance(stacked1, EagerTensor)
    assert stacked1.shape == (2, 3, 224, 224)

    stacked2 = TensorFlowCompBackend.stack([t0, t1], dim=-1)
    from tensorflow.python.framework.ops import EagerTensor

    assert isinstance(stacked2, EagerTensor)
    assert stacked2.shape == (3, 224, 224, 2)
