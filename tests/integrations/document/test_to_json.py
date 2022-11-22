import numpy as np
import torch

from docarray.document import BaseDocument
from docarray.typing import AnyUrl, Tensor, TorchTensor


def test_to_json():
    class Mmdoc(BaseDocument):
        img: Tensor
        url: AnyUrl
        txt: str
        torch_tensor: TorchTensor

    doc = Mmdoc(
        img=np.zeros((3, 224, 224)),
        url='http://doccaray.io',
        txt='hello',
        torch_tensor=torch.zeros(3, 224, 224),
    )
    doc.json()
