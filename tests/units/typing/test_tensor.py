import json

import numpy as np
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.typing import Tensor


def test_proto_tensor():

    tensor = parse_obj_as(Tensor, np.zeros((3, 224, 224)))

    tensor._to_node_protobuf()


def test_json_schema():
    schema_json_of(Tensor)


def test_dump_json(json_encoder):
    tensor = parse_obj_as(Tensor, np.zeros((3, 224, 224)))
    json.dumps(tensor, cls=json_encoder)
