import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray.base_document.io.json import orjson_dumps
from docarray.typing import TorchEmbedding, TorchTensor


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


def test_parametrized():
    # correct shape, single axis
    tensor = parse_obj_as(TorchTensor[128], torch.zeros(128))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (128,)

    # correct shape, multiple axis
    tensor = parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    # wrong but reshapable shape
    tensor = parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(224, 3, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    # wrong and not reshapable shape
    with pytest.raises(ValueError):
        parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(224, 224))

    tensor = parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    tensor = parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(3, 60, 128))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 60, 128)

    with pytest.raises(ValueError):
        parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(4, 224, 224))
    with pytest.raises(ValueError):
        parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(100, 1))

    tensor = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    with pytest.raises(ValueError):
        tensor = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 60, 128))


@pytest.mark.parametrize('shape', [(3, 224, 224), (224, 224, 3)])
def test_parameterized_tensor_class_name(shape):
    tensor = parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(shape))

    assert tensor.__class__.__name__ == 'TorchTensor[3, 224, 224]'
    assert tensor.__class__.__qualname__ == 'TorchTensor[3, 224, 224]'
    assert f'{tensor[0][0][0]}' == 'TorchTensor[3, 224, 224](0.)'


def test_torch_embedding():
    # correct shape
    tensor = parse_obj_as(TorchEmbedding[128], torch.zeros(128))
    assert isinstance(tensor, TorchEmbedding)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (128,)

    # wrong shape at data setting time
    with pytest.raises(ValueError):
        parse_obj_as(TorchEmbedding[128], torch.zeros(256))

    # illegal shape at class creation time
    with pytest.raises(ValueError):
        parse_obj_as(TorchEmbedding[128, 128], torch.zeros(128, 128))
