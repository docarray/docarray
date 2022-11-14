import numpy as np
from pydantic.tools import parse_obj_as

from docarray.typing import Tensor


def test_proto_tensor():

    uri = parse_obj_as(Tensor, np.zeros((3, 224, 224)))

    uri._to_node_protobuf()
