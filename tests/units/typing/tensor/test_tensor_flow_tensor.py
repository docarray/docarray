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
import numpy as np
import pytest
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.base_doc.io.json import orjson_dumps
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()
if tf_available:
    import tensorflow as tf
    import tensorflow._api.v2.experimental.numpy as tnp  # type: ignore
    from tensorflow.python.framework.errors_impl import InvalidArgumentError

    from docarray.typing import TensorFlowTensor


@pytest.mark.tensorflow
def test_proto_tensor():
    from docarray.proto.pb2.docarray_pb2 import NdArrayProto

    tensor = parse_obj_as(TensorFlowTensor, tf.zeros((3, 224, 224)))
    proto = tensor.to_protobuf()
    assert isinstance(proto, NdArrayProto)

    from_proto = TensorFlowTensor.from_protobuf(proto)
    assert isinstance(from_proto, TensorFlowTensor)
    assert tnp.allclose(tensor.tensor, from_proto.tensor)


@pytest.mark.tensorflow
def test_json_schema():
    schema_json_of(TensorFlowTensor)


@pytest.mark.tensorflow
def test_dump_json():
    tensor = parse_obj_as(TensorFlowTensor, tf.zeros((3, 224, 224)))
    orjson_dumps(tensor)


@pytest.mark.tensorflow
def test_unwrap():
    tf_tensor = parse_obj_as(TensorFlowTensor, tf.zeros((3, 224, 224)))
    unwrapped = tf_tensor.unwrap()

    assert not isinstance(unwrapped, TensorFlowTensor)
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(unwrapped, tf.Tensor)

    assert np.allclose(unwrapped, np.zeros((3, 224, 224)))


@pytest.mark.tensorflow
def test_from_ndarray():
    nd = np.array([1, 2, 3])
    tensor = TensorFlowTensor.from_ndarray(nd)
    assert isinstance(tensor, TensorFlowTensor)
    assert isinstance(tensor.tensor, tf.Tensor)


@pytest.mark.tensorflow
def test_ellipsis_in_shape():
    # ellipsis in the end, two extra dimensions needed
    tf_tensor = parse_obj_as(TensorFlowTensor[3, ...], tf.zeros((3, 128, 224)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 128, 224)

    # ellipsis in the beginning, two extra dimensions needed
    tf_tensor = parse_obj_as(TensorFlowTensor[..., 224], tf.zeros((3, 128, 224)))
    assert isinstance(tf_tensor, TensorFlowTensor)
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 128, 224)

    # more than one ellipsis in the shape
    with pytest.raises(ValueError):
        parse_obj_as(TensorFlowTensor[3, ..., 128, ...], tf.zeros((3, 128, 224)))

    # wrong shape
    with pytest.raises(ValueError):
        parse_obj_as(TensorFlowTensor[3, 224, ...], tf.zeros((3, 128, 224)))


@pytest.mark.tensorflow
def test_parametrized():
    # correct shape, single axis
    tf_tensor = parse_obj_as(TensorFlowTensor[128], tf.zeros(128))
    assert isinstance(tf_tensor, TensorFlowTensor)
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
    assert isinstance(tf_tensor.tensor, tf.Tensor)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    # wrong and not reshapable shape
    with pytest.raises(InvalidArgumentError):
        parse_obj_as(TensorFlowTensor[3, 224, 224], tf.zeros((224, 224)))


@pytest.mark.tensorflow
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


@pytest.mark.tensorflow
@pytest.mark.parametrize('shape', [(3, 224, 224), (224, 224, 3)])
def test_parameterized_tensor_class_name(shape):
    MyTFT = TensorFlowTensor[3, 224, 224]
    tensor = parse_obj_as(MyTFT, tf.zeros(shape))

    assert MyTFT.__name__ == 'TensorFlowTensor[3, 224, 224]'
    assert MyTFT.__qualname__ == 'TensorFlowTensor[3, 224, 224]'

    assert tensor.__class__.__name__ == 'TensorFlowTensor'
    assert tensor.__class__.__qualname__ == 'TensorFlowTensor'
    assert f'{tensor.tensor[0][0][0]}' == '0.0'


@pytest.mark.tensorflow
def test_parametrized_subclass():
    c1 = TensorFlowTensor[128]
    c2 = TensorFlowTensor[128]
    assert issubclass(c1, c2)
    assert issubclass(c1, TensorFlowTensor)

    assert not issubclass(c1, TensorFlowTensor[256])


@pytest.mark.tensorflow
def test_parametrized_instance():
    t = parse_obj_as(TensorFlowTensor[128], tf.zeros((128,)))
    assert isinstance(t, TensorFlowTensor[128])
    assert isinstance(t, TensorFlowTensor)
    # assert isinstance(t, tf.Tensor)

    assert not isinstance(t, TensorFlowTensor[256])
    assert not isinstance(t, TensorFlowTensor[2, 128])
    assert not isinstance(t, TensorFlowTensor[2, 2, 64])


@pytest.mark.tensorflow
def test_parametrized_equality():
    t1 = parse_obj_as(TensorFlowTensor[128], tf.zeros((128,)))
    t2 = parse_obj_as(TensorFlowTensor[128], tf.zeros((128,)))
    assert tf.experimental.numpy.allclose(t1.tensor, t2.tensor)


@pytest.mark.tensorflow
def test_parametrized_operations():
    t1 = parse_obj_as(TensorFlowTensor[128], tf.zeros((128,)))
    t2 = parse_obj_as(TensorFlowTensor[128], tf.zeros((128,)))
    t_result = t1.tensor + t2.tensor
    assert isinstance(t_result, tf.Tensor)
    assert not isinstance(t_result, TensorFlowTensor)
    assert not isinstance(t_result, TensorFlowTensor[128])


@pytest.mark.tensorflow
def test_set_item():
    t = TensorFlowTensor(tensor=tf.zeros((3, 224, 224)))
    t[0] = tf.ones((1, 224, 224))
    assert tnp.allclose(t.tensor[0], tf.ones((1, 224, 224)))
    assert tnp.allclose(t.tensor[1], tf.zeros((1, 224, 224)))
    assert tnp.allclose(t.tensor[2], tf.zeros((1, 224, 224)))
