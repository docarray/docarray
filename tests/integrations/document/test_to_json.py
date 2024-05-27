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

from docarray import BaseDoc, DocList
from docarray.base_doc.io.json import orjson_dumps
from docarray.typing import AnyUrl, NdArray, TorchTensor
from docarray.typing.bytes.image_bytes import ImageBytes


@pytest.fixture()
def doc_and_class():
    class Mmdoc(BaseDoc):
        img: NdArray
        url: AnyUrl
        txt: str
        torch_tensor: TorchTensor
        bytes_: bytes
        image_bytes_: ImageBytes

    doc = Mmdoc(
        img=np.zeros((10)),
        url='http://doccaray.io',
        txt='hello',
        torch_tensor=torch.zeros(10),
        bytes_=b'hello',
        image_bytes_=b'hello',
    )
    return doc, Mmdoc


def test_to_json(doc_and_class):
    import json

    doc, _ = doc_and_class
    js = doc.json()
    assert (
        json.loads(js)["image_bytes_"]
        == ImageBytes(b'hello')._docarray_to_json_compatible()
    )
    assert json.loads(js)["bytes_"] == 'hello'

    to_js = doc.to_json()
    assert (
        json.loads(to_js)["image_bytes_"]
        == ImageBytes(b'hello')._docarray_to_json_compatible()
    )
    assert json.loads(to_js)["bytes_"] == 'hello'


def test_doclist_to_json(doc_and_class):
    import json

    doc, cls = doc_and_class
    doc_list = DocList[cls]([doc, doc])
    js = doc_list.to_json()
    for d in json.loads(js):
        assert d["image_bytes_"] == ImageBytes(b'hello')._docarray_to_json_compatible()
        assert d["bytes_"] == 'hello'

    to_js = doc_list.to_json()
    for d in json.loads(to_js):
        assert d["image_bytes_"] == ImageBytes(b'hello')._docarray_to_json_compatible()
        assert d["bytes_"] == 'hello'


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
