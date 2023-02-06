import numpy as np
import orjson
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.base_document.io.json import orjson_dumps
from docarray.typing import NdArray
from docarray.typing.tensor import NdArrayEmbedding


@pytest.mark.proto
def test_proto_tensor():

    tensor = parse_obj_as(NdArray, np.zeros((3, 224, 224)))

    tensor._to_node_protobuf()


def test_from_list():
    tensor = parse_obj_as(NdArray, [[0.0, 0.0], [0.0, 0.0]])

    assert (tensor == np.zeros((2, 2))).all()


def test_json_schema():
    schema_json_of(NdArray)


def test_dump_json():
    tensor = parse_obj_as(NdArray, np.zeros((3, 224, 224)))
    orjson_dumps(tensor)


def test_load_json():
    tensor = parse_obj_as(NdArray, np.zeros((2, 2)))

    json = orjson_dumps(tensor)
    print(json)
    print(type(json))
    new_tensor = orjson.loads(json)

    assert (new_tensor == tensor).all()


def test_unwrap():
    tensor = parse_obj_as(NdArray, np.zeros((3, 224, 224)))
    ndarray = tensor.unwrap()

    assert not isinstance(ndarray, NdArray)
    assert isinstance(ndarray, np.ndarray)
    assert isinstance(tensor, NdArray)
    assert (ndarray == np.zeros((3, 224, 224))).all()


def test_parametrized():
    # correct shape, single axis
    tensor = parse_obj_as(NdArray[128], np.zeros(128))
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (128,)

    # correct shape, multiple axis
    tensor = parse_obj_as(NdArray[3, 224, 224], np.zeros((3, 224, 224)))
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 224, 224)

    # wrong but reshapable shape
    tensor = parse_obj_as(NdArray[3, 224, 224], np.zeros((3, 224, 224)))
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 224, 224)

    # wrong and not reshapable shape
    with pytest.raises(ValueError):
        parse_obj_as(NdArray[3, 224, 224], np.zeros((224, 224)))

    # test independent variable dimensions
    tensor = parse_obj_as(NdArray[3, 'x', 'y'], np.zeros((3, 224, 224)))
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 224, 224)

    tensor = parse_obj_as(NdArray[3, 'x', 'y'], np.zeros((3, 60, 128)))
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 60, 128)

    with pytest.raises(ValueError):
        parse_obj_as(NdArray[3, 'x', 'y'], np.zeros((4, 224, 224)))

    with pytest.raises(ValueError):
        parse_obj_as(NdArray[3, 'x', 'y'], np.zeros((100, 1)))

    # test dependent variable dimensions
    tensor = parse_obj_as(NdArray[3, 'x', 'x'], np.zeros((3, 224, 224)))
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (3, 224, 224)

    with pytest.raises(ValueError):
        tensor = parse_obj_as(NdArray[3, 'x', 'x'], np.zeros((3, 60, 128)))

    with pytest.raises(ValueError):
        tensor = parse_obj_as(NdArray[3, 'x', 'x'], np.zeros((3, 60)))


def test_np_embedding():
    # correct shape
    tensor = parse_obj_as(NdArrayEmbedding[128], np.zeros((128,)))
    assert isinstance(tensor, NdArrayEmbedding)
    assert isinstance(tensor, NdArray)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (128,)

    # wrong shape at data setting time
    with pytest.raises(ValueError):
        parse_obj_as(NdArrayEmbedding[128], np.zeros((256,)))

    # illegal shape at class creation time
    with pytest.raises(ValueError):
        parse_obj_as(NdArrayEmbedding[128, 128], np.zeros((128, 128)))


def test_parametrized_subclass():
    c1 = NdArray[128]
    c2 = NdArray[128]
    assert issubclass(c1, c2)
    assert issubclass(c1, NdArray)
    assert issubclass(c1, np.ndarray)

    assert not issubclass(c1, NdArray[256])


def test_parametrized_instance():
    t = parse_obj_as(NdArray[128], np.zeros(128))
    assert isinstance(t, NdArray[128])
    assert isinstance(t, NdArray)
    assert isinstance(t, np.ndarray)

    assert not isinstance(t, NdArray[256])
    assert not isinstance(t, NdArray[2, 64])
    assert not isinstance(t, NdArray[2, 2, 32])


def test_parametrized_equality():
    t1 = parse_obj_as(NdArray[128], np.zeros(128))
    t2 = parse_obj_as(NdArray[128], np.zeros(128))
    t3 = parse_obj_as(NdArray[256], np.zeros(256))
    assert (t1 == t2).all()
    assert not t1 == t3


def test_parametrized_operations():
    t1 = parse_obj_as(NdArray[128], np.zeros(128))
    t2 = parse_obj_as(NdArray[128], np.zeros(128))
    t_result = t1 + t2
    assert isinstance(t_result, np.ndarray)
    assert isinstance(t_result, NdArray)
    assert isinstance(t_result, NdArray[128])
