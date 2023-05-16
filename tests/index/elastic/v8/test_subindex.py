import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray
from tests.index.elastic.fixture import start_storage_v8  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index, pytest.mark.elasticv8]


class SimpleDoc(BaseDoc):
    simple_tens: NdArray[10] = Field(similarity='l2_norm')
    simple_text: str


class ListDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    simple_doc: SimpleDoc
    list_tens: NdArray[20] = Field(similarity='l2_norm')


class MyDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    list_docs: DocList[ListDoc]
    my_tens: NdArray[30] = Field(similarity='l2_norm')


@pytest.fixture
def index():
    index = ElasticDocIndex[MyDoc](index_name='idx')
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

    index.index(my_docs)
    return index


def test_subindex_init(index):
    assert isinstance(index._subindices['docs'], ElasticDocIndex)
    assert isinstance(index._subindices['list_docs'], ElasticDocIndex)
    assert isinstance(
        index._subindices['list_docs']._subindices['docs'], ElasticDocIndex
    )


def test_subindex_index(index):
    assert index.num_docs() == 5
    assert index._subindices['docs'].num_docs() == 25
    assert index._subindices['list_docs'].num_docs() == 25
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 125


def test_subindex_get(index):
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


def test_find_subindex(index):
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
    for root_doc, doc, score in zip(root_docs, docs, scores):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("-")[1]}'
        assert score == 1.0

    # sub sub level
    query = np.ones((10,))
    root_docs, docs, scores = index.find_subindex(
        query, subindex='list_docs__docs', search_field='simple_tens', limit=5
    )
    assert len(docs) == 5
    assert type(root_docs[0]) == MyDoc
    assert type(docs[0]) == SimpleDoc
    for root_doc, doc, score in zip(root_docs, docs, scores):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("-")[2]}'
        assert score == 1.0


def test_subindex_filter(index):
    query = {'match': {'simple_doc__simple_text': 'hello 0'}}
    docs = index.filter_subindex(query, subindex='list_docs', limit=5)
    assert len(docs) == 5
    assert type(docs[0]) == ListDoc
    for doc in docs:
        assert doc.id.split('-')[-1] == '0'

    query = {'match': {'simple_text': 'hello 0'}}
    docs = index.filter_subindex(query, subindex='list_docs__docs', limit=5)
    assert len(docs) == 5
    assert type(docs[0]) == SimpleDoc
    for doc in docs:
        assert doc.id.split('-')[-1] == '0'


def test_subindex_del(index):
    del index['0']
    assert index.num_docs() == 4
    assert index._subindices['docs'].num_docs() == 20
    assert index._subindices['list_docs'].num_docs() == 20
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 100
