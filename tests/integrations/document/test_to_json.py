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
import numpy as np
import pytest
import torch

from docarray.base_doc import BaseDoc
from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import AnyUrl, NdArray, TorchTensor


@pytest.fixture()
def doc_and_class():
    class Mmdoc(BaseDoc):
        img: NdArray
        url: AnyUrl
        txt: str
        torch_tensor: TorchTensor
        bytes_: bytes

    doc = Mmdoc(
        img=np.zeros((10)),
        url='http://doccaray.io',
        txt='hello',
        torch_tensor=torch.zeros(10),
        bytes_=b'hello',
    )
    return doc, Mmdoc


def test_to_json(doc_and_class):
    doc, _ = doc_and_class
    doc.json()


def test_from_json(doc_and_class):
    doc, Mmdoc = doc_and_class
    new_doc = Mmdoc.parse_raw(doc.json())

    for field, field2 in zip(doc.dict().keys(), new_doc.dict().keys()):
        if field in ['torch_tensor', 'img']:
            assert (getattr(doc, field) == getattr(doc, field2)).all()
        else:
            assert getattr(doc, field) == getattr(doc, field2)


def test_to_dict_to_json(doc_and_class):
    doc, Mmdoc = doc_and_class
    new_doc = Mmdoc.parse_raw(orjson_dumps(doc.dict()))

    for field, field2 in zip(doc.dict().keys(), new_doc.dict().keys()):
        if field in ['torch_tensor', 'img']:
            assert (getattr(doc, field) == getattr(doc, field2)).all()
        else:
            assert getattr(doc, field) == getattr(doc, field2)
