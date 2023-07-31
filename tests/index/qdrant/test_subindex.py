import numpy as np
import pytest
from pydantic import Field
from qdrant_client.http import models as rest

from docarray import BaseDoc, DocList
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import start_storage  # noqa: F401

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
def index():
    index = QdrantDocumentIndex[MyDoc](QdrantDocumentIndex.DBConfig(host='localhost'))
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
    assert isinstance(index._subindices['docs'], QdrantDocumentIndex)
    assert isinstance(index._subindices['list_docs'], QdrantDocumentIndex)
    assert isinstance(
        index._subindices['list_docs']._subindices['docs'], QdrantDocumentIndex
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
    for d in doc.docs:
        i = int(d.id.split('-')[-1])
        assert d.id == f'docs-1-{i}'
        assert np.allclose(d.simple_tens, np.ones(10) * (i + 1))

    assert len(doc.list_docs) == 5
    assert type(doc.list_docs[0]) == ListDoc
    assert set([d.id for d in doc.list_docs]) == set(
        [f'list_docs-1-{i}' for i in range(5)]
    )
    assert len(doc.list_docs[0].docs) == 5
    assert type(doc.list_docs[0].docs[0]) == SimpleDoc
    i = int(doc.list_docs[0].docs[0].id.split('-')[-2])
    j = int(doc.list_docs[0].docs[0].id.split('-')[-1])
    assert doc.list_docs[0].docs[0].id == f'list_docs-docs-1-{i}-{j}'
    assert np.allclose(doc.list_docs[0].docs[0].simple_tens, np.ones(10) * (j + 1))
    assert doc.list_docs[0].docs[0].simple_text == f'hello {j}'
    assert type(doc.list_docs[0].simple_doc) == SimpleDoc
    assert doc.list_docs[0].simple_doc.id == f'list_docs-simple_doc-1-{i}'
    assert np.allclose(doc.list_docs[0].simple_doc.simple_tens, np.ones(10) * (i + 1))
    assert doc.list_docs[0].simple_doc.simple_text == f'hello {i}'
    assert np.allclose(doc.list_docs[0].list_tens, np.ones(20) * (i + 1))

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
        assert score == 0.0

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
        assert score == 0.0


def test_subindex_filter(index):
    query = rest.Filter(
        must=[
            rest.FieldCondition(
                key='simple_doc__simple_text',
                match=rest.MatchText(text='hello 0'),
            )
        ]
    )
    docs = index.filter_subindex(query, subindex='list_docs', limit=5)
    assert len(docs) == 5
    assert type(docs[0]) == ListDoc
    for doc in docs:
        assert doc.id.split('-')[-1] == '0'

    query = rest.Filter(
        must=[
            rest.FieldCondition(
                key='simple_text',
                match=rest.MatchText(text='hello 0'),
            )
        ]
    )
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


def test_subindex_contain(index):
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
    empty_index = QdrantDocumentIndex[MyDoc]()
    assert (empty_doc in empty_index) is False


def test_subindex_collections():
    from typing import Optional
    from docarray.typing.tensor import AnyTensor
    from pydantic import Field

    class MetaPathDoc(BaseDoc):
        path_id: str
        level: int
        text: str
        embedding: Optional[AnyTensor] = Field(space='cosine', dim=128)

    class MetaCategoryDoc(BaseDoc):
        node_id: Optional[str]
        node_name: Optional[str]
        name: Optional[str]
        product_type_definitions: Optional[str]
        leaf: bool
        paths: Optional[DocList[MetaPathDoc]]
        embedding: Optional[AnyTensor] = Field(space='cosine', dim=128)
        channel: str
        lang: str

    db_config = QdrantDocumentIndex.DBConfig(
        host='localhost',
        collection_name="channel_category",
    )

    doc_index = QdrantDocumentIndex[MetaCategoryDoc](db_config)

    assert doc_index._subindices["paths"].index_name == 'channel_category__paths'
    assert doc_index._subindices["paths"].collection_name == 'channel_category__paths'
