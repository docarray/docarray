import numpy as np
from pydantic.tools import parse_obj_as

from docarray.typing import Embedding


def test_proto_embedding():

    uri = parse_obj_as(Embedding, np.zeros((3, 224, 224)))

    uri._to_node_protobuf()
