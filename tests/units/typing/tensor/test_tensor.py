import numpy as np
import orjson
import pytest
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document_base.io.json import orjson_dumps
from docarray.typing import NdArray
from docarray.typing.tensor import NdArrayEmbedding


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
