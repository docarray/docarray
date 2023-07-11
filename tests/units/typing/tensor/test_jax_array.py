import numpy as np
import pytest
from pydantic import schema_json_of
from pydantic.tools import parse_obj_as

from docarray.base_doc.io.json import orjson_dumps
from docarray.utils._internal.misc import is_jax_available

jax_available = is_jax_available()
if jax_available:
    import jax.numpy as jnp
    from jax._src.core import InconclusiveDimensionOperation

    from docarray.typing import JaxArray


@pytest.mark.jax
def test_proto_tensor():
    from docarray.proto.pb2.docarray_pb2 import NdArrayProto

    tensor = parse_obj_as(JaxArray, jnp.zeros((3, 224, 224)))
    proto = tensor.to_protobuf()
    assert isinstance(proto, NdArrayProto)

    from_proto = JaxArray.from_protobuf(proto)
    assert isinstance(from_proto, JaxArray)
    assert jnp.allclose(tensor.tensor, from_proto.tensor)


@pytest.mark.jax
def test_json_schema():
    schema_json_of(JaxArray)


@pytest.mark.jax
def test_dump_json():
    tensor = parse_obj_as(JaxArray, jnp.zeros((3, 224, 224)))
    orjson_dumps(tensor)


@pytest.mark.jax
def test_unwrap():
    tf_tensor = parse_obj_as(JaxArray, jnp.zeros((3, 224, 224)))
    unwrapped = tf_tensor.unwrap()

    assert not isinstance(unwrapped, JaxArray)
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(unwrapped, jnp.ndarray)

    assert np.allclose(unwrapped, np.zeros((3, 224, 224)))


@pytest.mark.jax
def test_from_ndarray():
    nd = np.array([1, 2, 3])
    tensor = JaxArray.from_ndarray(nd)
    assert isinstance(tensor, JaxArray)
    assert isinstance(tensor.tensor, jnp.ndarray)


@pytest.mark.jax
def test_ellipsis_in_shape():
    # ellipsis in the end, two extra dimensions needed
    tf_tensor = parse_obj_as(JaxArray[3, ...], jnp.zeros((3, 128, 224)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 128, 224)

    # ellipsis in the beginning, two extra dimensions needed
    tf_tensor = parse_obj_as(JaxArray[..., 224], jnp.zeros((3, 128, 224)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 128, 224)

    # more than one ellipsis in the shape
    with pytest.raises(ValueError):
        parse_obj_as(JaxArray[3, ..., 128, ...], jnp.zeros((3, 128, 224)))

    # wrong shape
    with pytest.raises(ValueError):
        parse_obj_as(JaxArray[3, 224, ...], jnp.zeros((3, 128, 224)))


@pytest.mark.jax
def test_parametrized():
    # correct shape, single axis
    tf_tensor = parse_obj_as(JaxArray[128], jnp.zeros(128))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (128,)

    # correct shape, multiple axis
    tf_tensor = parse_obj_as(JaxArray[3, 224, 224], jnp.zeros((3, 224, 224)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    # wrong but reshapable shape
    tf_tensor = parse_obj_as(JaxArray[3, 224, 224], jnp.zeros((224, 3, 224)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    # wrong and not reshapable shape
    with pytest.raises(InconclusiveDimensionOperation):
        parse_obj_as(JaxArray[3, 224, 224], jnp.zeros((224, 224)))


@pytest.mark.jax
def test_parametrized_with_str():
    # test independent variable dimensions
    tf_tensor = parse_obj_as(JaxArray[3, 'x', 'y'], jnp.zeros((3, 224, 224)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    tf_tensor = parse_obj_as(JaxArray[3, 'x', 'y'], jnp.zeros((3, 60, 128)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 60, 128)

    with pytest.raises(ValueError):
        parse_obj_as(JaxArray[3, 'x', 'y'], jnp.zeros((4, 224, 224)))

    with pytest.raises(ValueError):
        parse_obj_as(JaxArray[3, 'x', 'y'], jnp.zeros((100, 1)))

    # test dependent variable dimensions
    tf_tensor = parse_obj_as(JaxArray[3, 'x', 'x'], jnp.zeros((3, 224, 224)))
    assert isinstance(tf_tensor, JaxArray)
    assert isinstance(tf_tensor.tensor, jnp.ndarray)
    assert tf_tensor.tensor.shape == (3, 224, 224)

    with pytest.raises(ValueError):
        _ = parse_obj_as(JaxArray[3, 'x', 'x'], jnp.zeros((3, 60, 128)))

    with pytest.raises(ValueError):
        _ = parse_obj_as(JaxArray[3, 'x', 'x'], jnp.zeros((3, 60)))


@pytest.mark.jax
@pytest.mark.parametrize('shape', [(3, 224, 224), (224, 224, 3)])
def test_parameterized_tensor_class_name(shape):
    MyTFT = JaxArray[3, 224, 224]
    tensor = parse_obj_as(MyTFT, jnp.zeros(shape))

    assert MyTFT.__name__ == 'JaxArray[3, 224, 224]'
    assert MyTFT.__qualname__ == 'JaxArray[3, 224, 224]'

    assert tensor.__class__.__name__ == 'JaxArray'
    assert tensor.__class__.__qualname__ == 'JaxArray'
    assert f'{tensor.tensor[0][0][0]}' == '0.0'


@pytest.mark.jax
def test_parametrized_subclass():
    c1 = JaxArray[128]
    c2 = JaxArray[128]
    assert issubclass(c1, c2)
    assert issubclass(c1, JaxArray)

    assert not issubclass(c1, JaxArray[256])


@pytest.mark.jax
def test_parametrized_instance():
    t = parse_obj_as(JaxArray[128], jnp.zeros((128,)))
    assert isinstance(t, JaxArray[128])
    assert isinstance(t, JaxArray)
    # assert isinstance(t, jnp.ndarray)

    assert not isinstance(t, JaxArray[256])
    assert not isinstance(t, JaxArray[2, 128])
    assert not isinstance(t, JaxArray[2, 2, 64])


@pytest.mark.jax
def test_parametrized_equality():
    t1 = parse_obj_as(JaxArray[128], jnp.zeros((128,)))
    t2 = parse_obj_as(JaxArray[128], jnp.zeros((128,)))
    assert jnp.allclose(t1.tensor, t2.tensor)


@pytest.mark.jax
def test_parametrized_operations():
    t1 = parse_obj_as(JaxArray[128], jnp.zeros((128,)))
    t2 = parse_obj_as(JaxArray[128], jnp.zeros((128,)))
    t_result = t1.tensor + t2.tensor
    assert isinstance(t_result, jnp.ndarray)
    assert not isinstance(t_result, JaxArray)
    assert not isinstance(t_result, JaxArray[128])


@pytest.mark.jax
def test_set_item():
    t = JaxArray(tensor=jnp.zeros((3, 224, 224)))
    t[0] = jnp.ones((1, 224, 224))
    assert jnp.allclose(t.tensor[0], jnp.ones((1, 224, 224)))
    assert jnp.allclose(t.tensor[1], jnp.zeros((1, 224, 224)))
    assert jnp.allclose(t.tensor[2], jnp.zeros((1, 224, 224)))
