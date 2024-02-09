// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import pytest
import torch
from pydantic.tools import parse_obj_as, schema_json_of

from docarray import BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.proto import DocProto
from docarray.typing import TorchEmbedding, TorchTensor


class MyDoc(BaseDoc):
    tens: TorchTensor


@pytest.mark.proto
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


def test_parametrized_correct_axis_shape():
    # correct shape, single axis
    tensor = parse_obj_as(TorchTensor[128], torch.zeros(128))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (128,)


def test_correct_shape_multiple_axis():
    # correct shape, multiple axis
    tensor = parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_wrong_but_reshapable():
    # wrong but reshapable shape
    tensor = parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(224, 3, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    # wrong and not reshapable shape
    with pytest.raises(ValueError):
        parse_obj_as(TorchTensor[3, 224, 224], torch.zeros(224, 224))


def test_independent_variable_dim():
    # test independent variable dimensions
    tensor = parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_param():
    tensor = parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(3, 60, 128))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 60, 128)

    with pytest.raises(ValueError):
        parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(4, 224, 224))

    with pytest.raises(ValueError):
        parse_obj_as(TorchTensor[3, 'x', 'y'], torch.zeros(100, 1))


def test_dependent_variable_dim():
    # test dependent variable dimensions
    tensor = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 224, 224))
    assert isinstance(tensor, TorchTensor)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    with pytest.raises(ValueError):
        _ = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 60, 128))

    with pytest.raises(ValueError):
        _ = parse_obj_as(TorchTensor[3, 'x', 'x'], torch.zeros(3, 60))


@pytest.mark.parametrize('shape', [(3, 224, 224), (224, 224, 3)])
def test_parameterized_tensor_class_name(shape):
    MyTT = TorchTensor[3, 224, 224]
    tensor = parse_obj_as(MyTT, torch.zeros(shape))

    assert MyTT.__name__ == 'TorchTensor[3, 224, 224]'
    assert MyTT.__qualname__ == 'TorchTensor[3, 224, 224]'

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
    assert not isinstance(t, TorchTensor[2, 128])
    assert not isinstance(t, TorchTensor[2, 2, 64])


def test_parametrized_equality():
    t1 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t2 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    assert (t1 == t2).all()


def test_parametrized_operations():
    t1 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t2 = parse_obj_as(TorchTensor[128], torch.zeros(128))
    t_result = t1 + t2
    assert isinstance(t_result, torch.Tensor)
    assert isinstance(t_result, TorchTensor)
    assert isinstance(t_result, TorchTensor[128])


def test_deepcopy():
    from docarray import BaseDoc

    class MMdoc(BaseDoc):
        embedding: TorchEmbedding

    doc = MMdoc(embedding=torch.randn(32))
    doc_copy = doc.copy(deep=True)

    assert doc.embedding.data_ptr() != doc_copy.embedding.data_ptr()
    assert (doc.embedding == doc_copy.embedding).all()

    doc_copy.embedding = torch.randn(32)
    assert not (doc.embedding == doc_copy.embedding).all()


def test_deepcopy_tensor():
    from docarray import BaseDoc

    class MMdoc(BaseDoc):
        embedding: TorchTensor

    doc = MMdoc(embedding=torch.randn(32))
    doc_copy = doc.copy(deep=True)

    assert doc.embedding.data_ptr() != doc_copy.embedding.data_ptr()
    assert (doc.embedding == doc_copy.embedding).all()

    doc_copy.embedding = torch.randn(32)
    assert not (doc.embedding == doc_copy.embedding).all()


@pytest.mark.parametrize('requires_grad', [True])  # , False])
def test_json_serialization(requires_grad: bool):
    orig_doc = MyDoc(tens=torch.rand(10, requires_grad=requires_grad))
    serialized_doc = orig_doc.to_json()
    assert serialized_doc
    assert isinstance(serialized_doc, str)

    new_doc = MyDoc.from_json(serialized_doc)
    assert len(new_doc.tens) == 10


@pytest.mark.parametrize('protocol', ['pickle', 'protobuf'])
@pytest.mark.parametrize('requires_grad', [True, False])
def test_bytes_serialization(requires_grad, protocol):
    orig_doc = MyDoc(tens=torch.rand(10, requires_grad=requires_grad))
    serialized_doc = orig_doc.to_bytes(protocol=protocol)
    assert serialized_doc
    assert isinstance(serialized_doc, bytes)

    conv_doc = MyDoc.from_bytes(serialized_doc, protocol=protocol)
    assert isinstance(conv_doc.tens, TorchTensor)
    assert conv_doc.tens.shape == (10,)


@pytest.mark.parametrize('protocol', ['pickle', 'protobuf'])
@pytest.mark.parametrize('requires_grad', [True, False])
def test_base64_serialization(requires_grad, protocol):
    orig_doc = MyDoc(tens=torch.rand(10, requires_grad=requires_grad))
    serialized_doc = orig_doc.to_base64(protocol=protocol)
    assert serialized_doc
    assert isinstance(serialized_doc, str)

    conv_doc = MyDoc.from_base64(serialized_doc, protocol=protocol)
    assert isinstance(conv_doc.tens, TorchTensor)
    assert conv_doc.tens.shape == (10,)


@pytest.mark.parametrize('requires_grad', [True, False])
def test_protobuf_serialization(requires_grad: bool):
    orig_doc = MyDoc(tens=torch.rand(10, requires_grad=requires_grad))
    serialized_doc = orig_doc.to_protobuf()
    assert serialized_doc
    assert isinstance(serialized_doc, DocProto)

    conv_doc = MyDoc.from_protobuf(serialized_doc)
    assert isinstance(conv_doc.tens, TorchTensor)
    assert conv_doc.tens.shape == (10,)
