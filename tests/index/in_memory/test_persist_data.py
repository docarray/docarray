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
import pytest

from docarray import BaseDoc, DocList
from docarray.index import InMemoryExactNNIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    simple_tens: NdArray[10]
    simple_text: str


class ListDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    simple_doc: SimpleDoc
    list_tens: NdArray[20]


class MyDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    list_docs: DocList[ListDoc]
    my_tens: NdArray[30]


@pytest.fixture
def nested_doc():
    my_docs = [
        MyDoc(
            id=f'{i}',
            docs=DocList[SimpleDoc](
                [
                    SimpleDoc(
                        id=f'docs-{i}-{j}',
                        simple_tens=np.ones(10) * (j + 1),
                        simple_text=f'hello {j}',
                    )
                    for j in range(5)
                ]
            ),
            list_docs=DocList[ListDoc](
                [
                    ListDoc(
                        id=f'list_docs-{i}-{j}',
                        docs=DocList[SimpleDoc](
                            [
                                SimpleDoc(
                                    id=f'list_docs-docs-{i}-{j}-{k}',
                                    simple_tens=np.ones(10) * (k + 1),
                                    simple_text=f'hello {k}',
                                )
                                for k in range(5)
                            ]
                        ),
                        simple_doc=SimpleDoc(
                            id=f'list_docs-simple_doc-{i}-{j}',
                            simple_tens=np.ones(10) * (j + 1),
                            simple_text=f'hello {j}',
                        ),
                        list_tens=np.ones(20) * (j + 1),
                    )
                    for j in range(5)
                ]
            ),
            my_tens=np.ones((30,)) * (i + 1),
        )
        for i in range(5)
    ]
    return my_docs


def test_persist_restore(nested_doc, tmp_path):
    stored_path = str(tmp_path) + "/in_memory_index.bin"

    index = InMemoryExactNNIndex[MyDoc]()
    index.index(nested_doc)

    assert index.num_docs() == 5
    assert index._subindices['docs'].num_docs() == 25
    assert index._subindices['list_docs'].num_docs() == 25
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 125

    doc = index['1']

    assert type(doc.list_docs[0].simple_doc) == SimpleDoc
    assert doc.list_docs[0].simple_doc.id == 'list_docs-simple_doc-1-0'
    assert np.allclose(doc.list_docs[0].simple_doc.simple_tens, np.ones(10))
    assert doc.list_docs[0].simple_doc.simple_text == 'hello 0'

    del index['0']
    assert index.num_docs() == 4

    index.persist(stored_path)

    del index

    index = InMemoryExactNNIndex[MyDoc](index_file_path=stored_path)

    doc = index['1']

    assert index.num_docs() == 4
    assert index._subindices['docs'].num_docs() == 20
    assert index._subindices['list_docs'].num_docs() == 20
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 100
    assert type(doc) == MyDoc
    assert doc.list_docs[1].simple_doc.simple_text == 'hello 1'
    assert type(doc.list_docs[0].simple_doc) == SimpleDoc


def test_persist_find(nested_doc, tmp_path):
    index = InMemoryExactNNIndex[MyDoc]()
    index.index(nested_doc)

    stored_path = str(tmp_path) + "/in_memory_index.bin"
    index.persist(stored_path)

    del index
    index = InMemoryExactNNIndex[MyDoc](index_file_path=stored_path)

    # Test find
    query = np.ones((30,))
    docs, scores = index.find(query, search_field="my_tens", limit=5)

    assert type(docs[0]) == MyDoc
    assert type(docs[0].list_docs[0]) == ListDoc
    assert len(scores) == 5

    # Test find sub-index

    query = np.ones((10,))

    root_docs, docs, scores = index.find_subindex(
        query, subindex='docs', search_field='simple_tens', limit=5
    )

    assert type(root_docs[0]) == MyDoc
    assert type(docs[0]) == SimpleDoc
    assert len(scores) == 5
    for root_doc, doc in zip(root_docs, docs):
        assert root_doc.id == f'{doc.id.split("-")[1]}'
