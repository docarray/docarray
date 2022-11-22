import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.document.io.json import orjson_dumps
from docarray.typing import TorchTensor


def test_proto_tensor():

    tensor = parse_obj_as(TorchTensor, torch.zeros(3, 224, 224))

    tensor._to_node_protobuf()


def test_json_schema():
    schema_json_of(TorchTensor)


def test_dump_json():
    tensor = parse_obj_as(TorchTensor, torch.zeros(3, 224, 224))
    orjson_dumps(tensor)


def test_unwrap():
    tensor = parse_obj_as(TorchTensor, torch.zeros(3, 224, 224))
    ndarray = tensor.unwrap()

    assert not isinstance(ndarray, TorchTensor)
    assert isinstance(tensor, TorchTensor)
    assert isinstance(ndarray, torch.Tensor)

    assert tensor.data_ptr() == ndarray.data_ptr()

    assert (ndarray == torch.zeros(3, 224, 224)).all()
