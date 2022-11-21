import numpy as np
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps
from docarray.typing import Tensor


def test_proto_tensor():

    tensor = parse_obj_as(Tensor, np.zeros((3, 224, 224)))

    tensor._to_node_protobuf()


def test_json_schema():
    schema_json_of(Tensor)


def test_dump_json():
    tensor = parse_obj_as(Tensor, np.zeros((3, 224, 224)))
    orjson_dumps(tensor)


def test_unwrap():
    tensor = parse_obj_as(Tensor, np.zeros((3, 224, 224)))
    ndarray = tensor.unwrap()

    assert not isinstance(ndarray, Tensor)
    assert isinstance(ndarray, np.ndarray)
    assert isinstance(tensor, Tensor)
    assert (ndarray == np.zeros((3, 224, 224))).all()
