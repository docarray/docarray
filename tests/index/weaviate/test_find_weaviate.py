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
# TODO: enable ruff qa on this file when we figure out why it thinks weaviate_client is
#       redefined at each test that fixture
# ruff: noqa
import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc
from docarray.index.backends.weaviate import WeaviateDocumentIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.weaviate.fixture_weaviate import (  # noqa: F401
    start_storage,
    weaviate_client,
)

pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_find_torch(weaviate_client):
    class TorchDoc(BaseDoc):
        tens: TorchTensor[10] = Field(dims=10, is_embedding=True)

    index = WeaviateDocumentIndex[TorchDoc]()

    index_docs = [
        TorchDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    index.index(index_docs)

    query = index_docs[-1]
    docs, scores = index.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TorchTensor)

    assert docs[0].id == index_docs[-1].id
    assert torch.allclose(docs[0].tens, index_docs[-1].tens)


@pytest.mark.tensorflow
def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10] = Field(dims=10, is_embedding=True)

    index = WeaviateDocumentIndex[TfDoc]()

    index_docs = [
        TfDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    index.index(index_docs)

    query = index_docs[-1]
    docs, scores = index.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(
        docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )


def test_contain():
    class SimpleDoc(BaseDoc):
        tens: NdArray[10] = Field(dims=1000)

    class SimpleSchema(BaseDoc):
        tens: NdArray[10]

    index = WeaviateDocumentIndex[SimpleSchema]()
    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]

    assert (index_docs[0] in index) is False

    index.index(index_docs)

    for doc in index_docs:
        assert (doc in index) is True

    index_docs_new = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    for doc in index_docs_new:
        assert (doc in index) is False
