import numpy as np
import pytest
import tensorflow as tf
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.base_document.io.json import orjson_dumps
from docarray.typing import TensorFlowTensor


def test_json_schema():
    schema_json_of(TensorFlowTensor)


def test_dump_json():
    tensor = parse_obj_as(TensorFlowTensor, tf.zeros((3, 224, 224)))
    orjson_dumps(tensor)


def test_unwrap():
    tf_tensor = parse_obj_as(TensorFlowTensor, tf.zeros((3, 224, 224)))
    unwrapped = tf_tensor.unwrap()

    assert not isinstance(unwrapped, TensorFlowTensor)
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(unwrapped, tf.Tensor)

    assert np.allclose(unwrapped, np.zeros((3, 224, 224)))


def test_parametrized():
    # correct shape, single axis
    tf_tensor = parse_obj_as(TensorFlowTensor[128], tf.zeros(128))
    print(f"tf_tensor = {tf_tensor}")
    print(f"type(tf_tensor) = {type(tf_tensor)}")

    assert isinstance(tf_tensor, TensorFlowTensor)
    print(f"type(tf_tensor.tensor) = {type(tf_tensor.tensor)}")

    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (128,)

    # correct shape, multiple axis
    tf_tensor = parse_obj_as(TensorFlowTensor[3, 224, 224], tf.zeros((3, 224, 224)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    # wrong but reshapable shape
    tf_tensor = parse_obj_as(TensorFlowTensor[3, 224, 224], tf.zeros((224, 3, 224)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    # assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    # wrong and not reshapable shape
    from tensorflow.python.framework.errors_impl import InvalidArgumentError

    with pytest.raises(InvalidArgumentError):
        parse_obj_as(TensorFlowTensor[3, 224, 224], tf.zeros((224, 224)))


def test_parametrized_with_str():
    # test independent variable dimensions
    tf_tensor = parse_obj_as(TensorFlowTensor[3, 'x', 'y'], tf.zeros((3, 224, 224)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    tf_tensor = parse_obj_as(TensorFlowTensor[3, 'x', 'y'], tf.zeros((3, 60, 128)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 60, 128)

    with pytest.raises(ValueError):
        parse_obj_as(TensorFlowTensor[3, 'x', 'y'], tf.zeros((4, 224, 224)))

    with pytest.raises(ValueError):
        parse_obj_as(TensorFlowTensor[3, 'x', 'y'], tf.zeros((100, 1)))

    # test dependent variable dimensions
    tf_tensor = parse_obj_as(TensorFlowTensor[3, 'x', 'x'], tf.zeros((3, 224, 224)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    with pytest.raises(ValueError):
        _ = parse_obj_as(TensorFlowTensor[3, 'x', 'x'], tf.zeros((3, 60, 128)))

    with pytest.raises(ValueError):
        _ = parse_obj_as(TensorFlowTensor[3, 'x', 'x'], tf.zeros((3, 60)))
