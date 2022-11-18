import json

import numpy as np
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.typing import Embedding


def test_proto_embedding():

    embedding = parse_obj_as(Embedding, np.zeros((3, 224, 224)))

    embedding._to_node_protobuf()


def test_json_schema():
    schema_json_of(Embedding)


def test_dump_json(json_encoder):
    tensor = parse_obj_as(Embedding, np.zeros((3, 224, 224)))
    json.dumps(tensor, cls=json_encoder)
