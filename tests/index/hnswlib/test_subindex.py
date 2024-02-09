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
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    simple_tens: NdArray[10] = Field(space='l2')
    simple_text: str


class ListDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    simple_doc: SimpleDoc
    list_tens: NdArray[20] = Field(space='l2')


class MyDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    list_docs: DocList[ListDoc]
    my_tens: NdArray[30] = Field(space='l2')


@pytest.fixture(scope='session')
def index_docs():
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


def test_subindex_init(tmpdir, index_docs):
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.index(index_docs)
    assert isinstance(index._subindices['docs'], HnswDocumentIndex)
    assert isinstance(index._subindices['list_docs'], HnswDocumentIndex)
    assert isinstance(
        index._subindices['list_docs']._subindices['docs'], HnswDocumentIndex
    )


def test_subindex_index(tmpdir, index_docs):
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.index(index_docs)
    assert index.num_docs() == 5
    assert index._subindices['docs'].num_docs() == 25
    assert index._subindices['list_docs'].num_docs() == 25
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 125


def test_subindex_get(tmpdir, index_docs):
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.index(index_docs)
    doc = index['1']
    assert type(doc) == MyDoc
    assert doc.id == '1'
    assert len(doc.docs) == 5
    assert type(doc.docs[0]) == SimpleDoc
    assert doc.docs[0].id == 'docs-1-0'
    assert np.allclose(doc.docs[0].simple_tens, np.ones(10))

    assert len(doc.list_docs) == 5
    assert type(doc.list_docs[0]) == ListDoc
    assert doc.list_docs[0].id == 'list_docs-1-0'
    assert len(doc.list_docs[0].docs) == 5
    assert type(doc.list_docs[0].docs[0]) == SimpleDoc
    assert doc.list_docs[0].docs[0].id == 'list_docs-docs-1-0-0'
    assert np.allclose(doc.list_docs[0].docs[0].simple_tens, np.ones(10))
    assert doc.list_docs[0].docs[0].simple_text == 'hello 0'
    assert type(doc.list_docs[0].simple_doc) == SimpleDoc
    assert doc.list_docs[0].simple_doc.id == 'list_docs-simple_doc-1-0'
    assert np.allclose(doc.list_docs[0].simple_doc.simple_tens, np.ones(10))
    assert doc.list_docs[0].simple_doc.simple_text == 'hello 0'
    assert np.allclose(doc.list_docs[0].list_tens, np.ones(20))

    assert np.allclose(doc.my_tens, np.ones(30) * 2)


def test_find_subindex(tmpdir, index_docs):
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.index(index_docs)
    # root level
    query = np.ones((30,))
    with pytest.raises(ValueError):
        _, _ = index.find_subindex(query, subindex='', search_field='my_tens', limit=5)

    # sub level
    query = np.ones((10,))
    root_docs, docs, scores = index.find_subindex(
        query, subindex='docs', search_field='simple_tens', limit=5
    )
    assert type(root_docs[0]) == MyDoc
    assert type(docs[0]) == SimpleDoc
    assert len(scores) == 5
    for root_doc, doc in zip(root_docs, docs):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("-")[1]}'

    # sub sub level
    query = np.ones((10,))
    root_docs, docs, scores = index.find_subindex(
        query, subindex='list_docs__docs', search_field='simple_tens', limit=5
    )
    assert len(docs) == 5
    assert len(scores) == 5
    assert type(root_docs[0]) == MyDoc
    assert type(docs[0]) == SimpleDoc
    for root_doc, doc in zip(root_docs, docs):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("-")[2]}'


def test_subindex_del(tmpdir, index_docs):
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.index(index_docs)
    del index['0']
    assert index.num_docs() == 4
    assert index._subindices['docs'].num_docs() == 20
    assert index._subindices['list_docs'].num_docs() == 20
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 100


def test_subindex_contain(tmpdir, index_docs):
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.index(index_docs)
    # Checks for individual simple_docs within list_docs
    for i in range(4):
        doc = index[f'{i + 1}']
        for simple_doc in doc.list_docs:
            assert index.subindex_contains(simple_doc) is True
            for nested_doc in simple_doc.docs:
                assert index.subindex_contains(nested_doc) is True

    invalid_doc = SimpleDoc(
        id='non_existent',
        simple_tens=np.zeros(10),
        simple_text='invalid',
    )
    assert index.subindex_contains(invalid_doc) is False

    # Checks for an empty doc
    empty_doc = SimpleDoc(
        id='',
        simple_tens=np.zeros(10),
        simple_text='',
    )
    assert index.subindex_contains(empty_doc) is False

    # Empty index
    empty_index = HnswDocumentIndex[MyDoc]()
    assert (empty_doc in empty_index) is False
