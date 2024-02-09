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
from typing import Optional

import torch

from docarray import BaseDoc, DocList
from docarray.typing import TorchTensor


def test_torch_train():
    class Mmdoc(BaseDoc):
        text: str
        tensor: Optional[TorchTensor[3, 224, 224]] = None

    N = 10

    batch = DocList[Mmdoc](Mmdoc(text=f'hello{i}') for i in range(N))
    batch.tensor = torch.zeros(N, 3, 224, 224)

    batch = batch.to_doc_vec()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for _ in range(2):
        loss = model(batch.tensor).sum()
        loss.backward()
        opt.step()
