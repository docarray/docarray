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
import numpy as np

from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    embedding: NdArray[128]
    text: str


def test_update_payload():
    docs = DocList[SimpleDoc](
        [SimpleDoc(embedding=np.random.rand(128), text=f'hey {i}') for i in range(100)]
    )
    index = InMemoryExactNNIndex[SimpleDoc]()
    index.index(docs)

    assert index.num_docs() == 100

    for doc in docs:
        doc.text += '_changed'

    index.index(docs)
    assert index.num_docs() == 100

    res = index.find(query=docs[0], search_field='embedding', limit=100)
    assert len(res.documents) == 100
    for doc in res.documents:
        assert '_changed' in doc.text


def test_update_embedding():
    docs = DocList[SimpleDoc](
        [SimpleDoc(embedding=np.random.rand(128), text=f'hey {i}') for i in range(100)]
    )
    index = InMemoryExactNNIndex[SimpleDoc]()
    index.index(docs)
    assert index.num_docs() == 100

    new_tensor = np.random.rand(128)
    docs[0].embedding = new_tensor

    index.index(docs[0])
    assert index.num_docs() == 100

    res = index.find(query=docs[0], search_field='embedding', limit=100)
    assert len(res.documents) == 100
    found = False
    for doc in res.documents:
        if doc.id == docs[0].id:
            found = True
            assert (doc.embedding == new_tensor).all()
    assert found
