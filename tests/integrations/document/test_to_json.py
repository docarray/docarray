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


def test_from_json():
    class Mmdoc(BaseDocument):
        img: Tensor
        url: AnyUrl
        txt: str
        torch_tensor: TorchTensor

    doc = Mmdoc(
        img=np.zeros((2, 2)),
        url='http://doccaray.io',
        txt='hello',
        torch_tensor=torch.zeros(3, 224, 224),
    )
    new_doc = Mmdoc.parse_raw(doc.json())

    for (field, field2) in zip(doc.dict().keys(), new_doc.dict().keys()):
        if field in ['torch_tensor', 'img']:
            assert (getattr(doc, field) == getattr(doc, field2)).all()
        else:
            assert getattr(doc, field) == getattr(doc, field2)
