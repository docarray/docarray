from typing import Optional

import torch

from docarray import BaseDoc, DocList
from docarray.typing import TorchTensor


def test_torch_train():
    class Mmdoc(BaseDoc):
        text: str
        tensor: Optional[TorchTensor[3, 224, 224]]

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
