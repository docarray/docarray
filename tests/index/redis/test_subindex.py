import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray
from tests.index.redis.fixtures import start_redis  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    simple_tens: NdArray[10] = Field(space='l2')
    simple_text: str


class ListDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    simple_doc: SimpleDoc
    list_tens: NdArray[20] = Field(space='l2')


class NestedDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    list_docs: DocList[ListDoc]
    my_tens: NdArray[30] = Field(space='l2')


@pytest.fixture(scope='session')
def index():
    index = RedisDocumentIndex[NestedDoc](host='localhost')
    return index


@pytest.fixture(scope='session')
def data():
    my_docs = [
        NestedDoc(
            id=f'{i}',
            docs=DocList[SimpleDoc](
                [
                    SimpleDoc(
                        id=f'docs_{i}_{j}',
                        simple_tens=np.ones(10) * (j + 1),
                        simple_text=f'hello {j}',
                    )
                    for j in range(5)
                ]
            ),
            list_docs=DocList[ListDoc](
                [
                    ListDoc(
                        id=f'list_docs_{i}_{j}',
                        docs=DocList[SimpleDoc](
                            [
                                SimpleDoc(
                                    id=f'list_docs_docs_{i}_{j}_{k}',
                                    simple_tens=np.ones(10) * (k + 1),
                                    simple_text=f'hello {k}',
                                )
                                for k in range(5)
                            ]
                        ),
                        simple_doc=SimpleDoc(
                            id=f'list_docs_simple_doc_{i}_{j}',
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


def test_subindex_init(index):
    assert isinstance(index._subindices['docs'], RedisDocumentIndex)
    assert isinstance(index._subindices['list_docs'], RedisDocumentIndex)
    assert isinstance(
        index._subindices['list_docs']._subindices['docs'], RedisDocumentIndex
    )


def test_subindex_index(index, data):
    index.index(data)
    assert index.num_docs() == 5
    assert index._subindices['docs'].num_docs() == 25
    assert index._subindices['list_docs'].num_docs() == 25
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 125


def test_subindex_get(index, data):
    index.index(data)
    doc = index['1']
    assert type(doc) == NestedDoc
    assert doc.id == '1'
    assert len(doc.docs) == 5
    assert type(doc.docs[0]) == SimpleDoc
    assert doc.docs[0].id == 'docs_1_0'
    assert np.allclose(doc.docs[0].simple_tens, np.ones(10))

    assert len(doc.list_docs) == 5
    assert type(doc.list_docs[0]) == ListDoc
    assert doc.list_docs[0].id == 'list_docs_1_0'
    assert len(doc.list_docs[0].docs) == 5
    assert type(doc.list_docs[0].docs[0]) == SimpleDoc
    assert doc.list_docs[0].docs[0].id == 'list_docs_docs_1_0_0'
    assert np.allclose(doc.list_docs[0].docs[0].simple_tens, np.ones(10))
    assert doc.list_docs[0].docs[0].simple_text == 'hello 0'
    assert type(doc.list_docs[0].simple_doc) == SimpleDoc
    assert doc.list_docs[0].simple_doc.id == 'list_docs_simple_doc_1_0'
    assert np.allclose(doc.list_docs[0].simple_doc.simple_tens, np.ones(10))
    assert doc.list_docs[0].simple_doc.simple_text == 'hello 0'
    assert np.allclose(doc.list_docs[0].list_tens, np.ones(20))

    assert np.allclose(doc.my_tens, np.ones(30) * 2)


def test_subindex_del(index, data):
    index.index(data)
    del index['0']
    assert index.num_docs() == 4
    assert index._subindices['docs'].num_docs() == 20
    assert index._subindices['list_docs'].num_docs() == 20
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 100


def test_subindex_contain(index, data):
    index.index(data)
    # Checks for individual simple_docs within list_docs
    for i in range(4):
        doc = index[f'{i + 1}']
        for simple_doc in doc.list_docs:
            assert index.subindex_contains(simple_doc)
            for nested_doc in simple_doc.docs:
                assert index.subindex_contains(nested_doc)

    invalid_doc = SimpleDoc(
        id='non_existent',
        simple_tens=np.zeros(10),
        simple_text='invalid',
    )
    assert not index.subindex_contains(invalid_doc)

    # Checks for an empty doc
    empty_doc = SimpleDoc(
        id='',
        simple_tens=np.zeros(10),
        simple_text='',
    )
    assert not index.subindex_contains(empty_doc)

    # Empty index
    empty_index = RedisDocumentIndex[NestedDoc](host='localhost')
    assert empty_doc not in empty_index


def test_find_subindex(index, data):
    index.index(data)
    # root level
    query = np.ones((30,))
    with pytest.raises(ValueError):
        _, _ = index.find_subindex(query, subindex='', search_field='my_tens', limit=5)

    # sub level
    query = np.ones((10,))
    root_docs, docs, scores = index.find_subindex(
        query, subindex='docs', search_field='simple_tens', limit=5
    )
    assert type(root_docs[0]) == NestedDoc
    assert type(docs[0]) == SimpleDoc
    assert len(scores) == 5
    for root_doc, doc in zip(root_docs, docs):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("_")[-2]}'

    # sub sub level
    query = np.ones((10,))
    root_docs, docs, scores = index.find_subindex(
        query, subindex='list_docs__docs', search_field='simple_tens', limit=5
    )
    assert len(docs) == 5
    assert len(scores) == 5
    assert type(root_docs[0]) == NestedDoc
    assert type(docs[0]) == SimpleDoc
    for root_doc, doc in zip(root_docs, docs):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("_")[-3]}'
