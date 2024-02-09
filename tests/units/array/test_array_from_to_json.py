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
from typing import Optional, Dict, List

import numpy as np
import pytest
import torch

from docarray import BaseDoc, DocList, DocVec
from docarray.documents import ImageDoc
from docarray.typing import NdArray, TorchTensor


class MyDoc(BaseDoc):
    embedding: NdArray
    text: str
    image: ImageDoc


def test_from_to_json_doclist():
    da = DocList[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ]
    )
    json_da = da.to_json()
    da2 = DocList[MyDoc].from_json(json_da)
    assert len(da2) == 2
    assert len(da) == len(da2)
    for d1, d2 in zip(da, da2):
        assert d1.embedding.tolist() == d2.embedding.tolist()
        assert d1.text == d2.text
        assert d1.image.url == d2.image.url
    assert da[1].image.url is None
    assert da2[1].image.url is None


@pytest.mark.parametrize('tensor_type', [TorchTensor, NdArray])
def test_from_to_json_docvec(tensor_type):
    def generate_docs(tensor_type):
        class InnerDoc(BaseDoc):
            tens: tensor_type

        class MyDoc(BaseDoc):
            text: str
            num: Optional[int] = None
            tens: tensor_type
            tens_none: Optional[tensor_type] = None
            inner: InnerDoc
            inner_none: Optional[InnerDoc] = None
            inner_vec: DocVec[InnerDoc]
            inner_vec_none: Optional[DocVec[InnerDoc]] = None

        def _rand_vec_gen(tensor_type):
            arr = np.random.rand(5)
            if tensor_type == TorchTensor:
                arr = torch.from_numpy(arr).to(torch.float32)
            return arr

        inner = InnerDoc(tens=_rand_vec_gen(tensor_type))
        inner_vec = DocVec[InnerDoc]([inner, inner], tensor_type=tensor_type)
        vec = DocVec[MyDoc](
            [
                MyDoc(
                    text=str(i),
                    num=None,
                    tens=_rand_vec_gen(tensor_type),
                    inner=inner,
                    inner_none=None,
                    inner_vec=inner_vec,
                    inner_vec_none=None,
                )
                for i in range(5)
            ],
            tensor_type=tensor_type,
        )
        return vec

    v = generate_docs(tensor_type)
    json_str = v.to_json()

    v_after = DocVec[v.doc_type].from_json(json_str, tensor_type=tensor_type)

    assert v_after.tensor_type == v.tensor_type
    assert set(v_after._storage.columns.keys()) == set(v._storage.columns.keys())
    assert v_after._storage == v._storage


@pytest.mark.tensorflow
def test_from_to_json_docvec_tf():
    from docarray.typing import TensorFlowTensor

    def generate_docs():
        class InnerDoc(BaseDoc):
            tens: TensorFlowTensor

        class MyDoc(BaseDoc):
            text: str
            num: Optional[int] = None
            tens: TensorFlowTensor
            tens_none: Optional[TensorFlowTensor] = None
            inner: InnerDoc
            inner_none: Optional[InnerDoc] = None
            inner_vec: DocVec[InnerDoc]
            inner_vec_none: Optional[DocVec[InnerDoc]] = None

        inner = InnerDoc(tens=np.random.rand(5))
        inner_vec = DocVec[InnerDoc]([inner, inner], tensor_type=TensorFlowTensor)
        vec = DocVec[MyDoc](
            [
                MyDoc(
                    text=str(i),
                    num=None,
                    tens=np.random.rand(5),
                    inner=inner,
                    inner_none=None,
                    inner_vec=inner_vec,
                    inner_vec_none=None,
                )
                for i in range(5)
            ],
            tensor_type=TensorFlowTensor,
        )
        return vec

    v = generate_docs()
    json_str = v.to_json()

    v_after = DocVec[v.doc_type].from_json(json_str, tensor_type=TensorFlowTensor)

    assert v_after.tensor_type == v.tensor_type
    assert set(v_after._storage.columns.keys()) == set(v._storage.columns.keys())
    assert v_after._storage == v._storage


def test_union_type():
    from typing import Union

    from docarray.documents import TextDoc

    class CustomDoc(BaseDoc):
        ud: Union[TextDoc, ImageDoc] = TextDoc(text='union type')

    docs = DocList[CustomDoc]([CustomDoc(ud=TextDoc(text='union type'))])

    docs_copy = docs.from_json(docs.to_json())
    assert docs == docs_copy


@pytest.mark.parametrize('tensor_type', [NdArray, TorchTensor])
def test_from_to_json_tensor_type(tensor_type):
    da = DocVec[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ],
        tensor_type=tensor_type,
    )
    json_da = da.to_json()
    da2 = DocVec[MyDoc].from_json(json_da, tensor_type=tensor_type)
    assert da2.tensor_type == tensor_type
    assert isinstance(da2.embedding, tensor_type)
