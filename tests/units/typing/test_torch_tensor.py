import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.typing import TorchTensor


def test_proto_tensor():

    tensor = parse_obj_as(TorchTensor, torch.zeros(3, 224, 224))

    tensor._to_node_protobuf()


def test_json_schema():
    schema_json_of(TorchTensor)


# def test_dump_json(json_encoder):
#     tensor = parse_obj_as(Tensor, torch.zeros(3, 224, 224))
#     json.dumps(tensor, cls=json_encoder)


def test_unwrap():
    tensor = parse_obj_as(TorchTensor, torch.zeros(3, 224, 224))
    ndarray = tensor.unwrap()

    assert not isinstance(ndarray, TorchTensor)
    assert isinstance(tensor, TorchTensor)
    assert isinstance(ndarray, torch.Tensor)

    assert (ndarray == torch.zeros(3, 224, 224)).all()
