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

    # test independent variable dimensions
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

    # test dependent variable dimensions
    tensor = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    with pytest.raises(ValueError):
        tensor = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 60, 128))

    with pytest.raises(ValueError):
        tensor = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 60))


@pytest.mark.parametrize('shape', [(3, 224, 224), (224, 224, 3)])
def test_parameterized_tensor_class_name(shape):
    tensor = parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(shape))

    assert tensor.__class__.__name__ == 'TorchTensor'
    assert tensor.__class__.__qualname__ == 'TorchTensor'
    assert f'{tensor[0][0][0]}' == 'TorchTensor(0.)'


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


def test_parametrized_subclass():
    c1 = TorchTensor[128]
    c2 = TorchTensor[128]
    assert issubclass(c1, c2)
    assert issubclass(c1, TorchTensor)
    assert issubclass(c1, torch.Tensor)

    assert not issubclass(c1, TorchTensor[256])


def test_parametrized_instance():
    t = parse_obj_as(TorchTensor[128], torch.zeros(128))
    assert isinstance(t, TorchTensor[128])
    assert isinstance(t, TorchTensor)
    assert isinstance(t, torch.Tensor)

    assert not isinstance(t, TorchTensor[256])


def test_parametrized_equality():
    t1 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t2 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t3 = parse_obj_as(TorchTensor[256], torch.zeros(256))
    assert (t1 == t2).all()
    with pytest.raises(RuntimeError):
        t1 == t3


def test_parametrized_operations():
    t1 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t2 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t_result = t1 + t2
    assert isinstance(t_result, torch.Tensor)
    assert isinstance(t_result, TorchTensor)
    assert isinstance(t_result, TorchTensor[128])
