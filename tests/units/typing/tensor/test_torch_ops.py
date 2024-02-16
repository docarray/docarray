# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from docarray import BaseDoc
from docarray.typing import TorchTensor


def test_tensor_ops():
    class A(BaseDoc):
        tensor: TorchTensor[3, 224, 224]

    class B(BaseDoc):
        tensor: TorchTensor[3, 112, 224]

    tensor = A(tensor=torch.ones(3, 224, 224)).tensor
    tensord = A(tensor=torch.ones(3, 224, 224)).tensor
    tensorn = torch.zeros(3, 224, 224)
    tensorhalf = B(tensor=torch.ones(3, 112, 224)).tensor
    tensorfull = torch.cat([tensorhalf, tensorhalf], dim=1)

    assert type(tensor) == TorchTensor
    assert type(tensor + tensord) == TorchTensor
    assert type(tensor + tensorn) == TorchTensor
    assert type(tensor + tensorfull) == TorchTensor
