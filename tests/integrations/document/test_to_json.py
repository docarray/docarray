import numpy as np
import pytest
import torch

from docarray.document import BaseDocument
from docarray.typing import AnyUrl, NdArray, TorchTensor


@pytest.fixture()
def doc_and_class():
    class Mmdoc(BaseDocument):
        img: NdArray
        url: AnyUrl
        txt: str
        torch_tensor: TorchTensor

    doc = Mmdoc(
        img=np.zeros((3, 224, 224)),
        url='http://doccaray.io',
        txt='hello',
        torch_tensor=torch.zeros(3, 224, 224),
    )
    return doc, Mmdoc


def test_to_json(doc_and_class):
    doc, _ = doc_and_class
    doc.json()


def test_from_json(doc_and_class):
    doc, Mmdoc = doc_and_class
    new_doc = Mmdoc.parse_raw(doc.json())

    for (field, field2) in zip(doc.dict().keys(), new_doc.dict().keys()):
        if field in ['torch_tensor', 'img']:
            assert (getattr(doc, field) == getattr(doc, field2)).all()
        else:
            assert getattr(doc, field) == getattr(doc, field2)
