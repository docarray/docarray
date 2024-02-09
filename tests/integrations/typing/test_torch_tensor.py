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
import torch
from docarray.typing.tensor.torch_tensor import TorchTensor
import copy

from docarray import BaseDoc
from docarray.typing import TorchEmbedding, TorchTensor


def test_set_torch_tensor():
    class MyDocument(BaseDoc):
        tensor: TorchTensor

    d = MyDocument(tensor=torch.zeros((3, 224, 224)))

    assert isinstance(d.tensor, TorchTensor)
    assert isinstance(d.tensor, torch.Tensor)
    assert (d.tensor == torch.zeros((3, 224, 224))).all()


def test_set_torch_embedding():
    class MyDocument(BaseDoc):
        embedding: TorchEmbedding

    d = MyDocument(embedding=torch.zeros((128,)))

    assert isinstance(d.embedding, TorchTensor)
    assert isinstance(d.embedding, TorchEmbedding)
    assert isinstance(d.embedding, torch.Tensor)
    assert (d.embedding == torch.zeros((128,))).all()


def test_torchtensor_deepcopy():
    # Setup
    original_tensor_float = TorchTensor(torch.rand(10))
    original_tensor_int = TorchTensor(torch.randint(0, 100, (10,)))

    # Exercise
    copied_tensor_float = copy.deepcopy(original_tensor_float)
    copied_tensor_int = copy.deepcopy(original_tensor_int)

    # Verify
    assert torch.equal(original_tensor_float, copied_tensor_float)
    assert original_tensor_float.data_ptr() != copied_tensor_float.data_ptr()
    assert torch.equal(original_tensor_int, copied_tensor_int)
    assert original_tensor_int.data_ptr() != copied_tensor_int.data_ptr()
