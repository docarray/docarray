import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import MongoAtlasDocumentIndex
from docarray.typing import NdArray

from .fixtures import *  # noqa

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    simple_tens: NdArray[10] = Field(index_name='vector_index')
    simple_text: str


class ListDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    simple_doc: SimpleDoc
    list_tens: NdArray[20] = Field(space='l2')


class MyDoc(BaseDoc):
    docs: DocList[SimpleDoc]
    list_docs: DocList[ListDoc]
    my_tens: NdArray[30] = Field(space='l2')


def clean_subindex(index):
    for subindex in index._subindices.values():
        clean_subindex(subindex)
    index._doc_collection.delete_many({})


@pytest.fixture(scope='session')
def index(mongo_fixture_env):
    uri, database = mongo_fixture_env
    index = MongoAtlasDocumentIndex[MyDoc](
        mongo_connection_uri=uri,
        database_name=database,
    )
    clean_subindex(index)

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
                    for j in range(2)
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
                                for k in range(2)
                            ]
                        ),
                        simple_doc=SimpleDoc(
                            id=f'list_docs-simple_doc-{i}-{j}',
                            simple_tens=np.ones(10) * (j + 1),
                            simple_text=f'hello {j}',
                        ),
                        list_tens=np.ones(20) * (j + 1),
                    )
                    for j in range(2)
                ]
            ),
            my_tens=np.ones((30,)) * (i + 1),
        )
        for i in range(2)
    ]

    index.index(my_docs)
    yield index
    clean_subindex(index)


def test_subindex_init(index):
    assert isinstance(index._subindices['docs'], MongoAtlasDocumentIndex)
    assert isinstance(index._subindices['list_docs'], MongoAtlasDocumentIndex)
    assert isinstance(
        index._subindices['list_docs']._subindices['docs'], MongoAtlasDocumentIndex
    )


def test_subindex_index(index):
    assert index.num_docs() == 2
    assert index._subindices['docs'].num_docs() == 4
    assert index._subindices['list_docs'].num_docs() == 4
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 8


def test_subindex_get(index):
    doc = index['1']
    assert isinstance(doc, MyDoc)
    assert doc.id == '1'

    assert len(doc.docs) == 2
    assert isinstance(doc.docs[0], SimpleDoc)
    for d in doc.docs:
        i = int(d.id.split('-')[-1])
        assert d.id == f'docs-1-{i}'
        assert np.allclose(d.simple_tens, np.ones(10) * (i + 1))

    assert len(doc.list_docs) == 2
    assert isinstance(doc.list_docs[0], ListDoc)
    assert set([d.id for d in doc.list_docs]) == set(
        [f'list_docs-1-{i}' for i in range(2)]
    )
    assert len(doc.list_docs[0].docs) == 2
    assert isinstance(doc.list_docs[0].docs[0], SimpleDoc)
    i = int(doc.list_docs[0].docs[0].id.split('-')[-2])
    j = int(doc.list_docs[0].docs[0].id.split('-')[-1])
    assert doc.list_docs[0].docs[0].id == f'list_docs-docs-1-{i}-{j}'
    assert np.allclose(doc.list_docs[0].docs[0].simple_tens, np.ones(10) * (j + 1))
    assert doc.list_docs[0].docs[0].simple_text == f'hello {j}'
    assert isinstance(doc.list_docs[0].simple_doc, SimpleDoc)
    assert doc.list_docs[0].simple_doc.id == f'list_docs-simple_doc-1-{i}'
    assert np.allclose(doc.list_docs[0].simple_doc.simple_tens, np.ones(10) * (i + 1))
    assert doc.list_docs[0].simple_doc.simple_text == f'hello {i}'
    assert np.allclose(doc.list_docs[0].list_tens, np.ones(20) * (i + 1))

    assert np.allclose(doc.my_tens, np.ones(30) * 2)


def test_subindex_contain(index, mongo_fixture_env):
    # Checks for individual simple_docs within list_docs

    doc = index['0']
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
    uri, database = mongo_fixture_env
    empty_index = MongoAtlasDocumentIndex[MyDoc](
        mongo_connection_uri=uri,
        database_name="random_database",
    )
    assert (empty_doc in empty_index) is False


def test_find_empty_subindex(index):
    query = np.ones((30,))
    with pytest.raises(ValueError):
        index.find_subindex(query, subindex='', search_field='my_tens', limit=5)


def test_find_subindex_sublevel(index):
    query = np.ones((10,))

    root_docs, docs, scores = index.find_subindex(
        query, subindex='docs', search_field='simple_tens', limit=4
    )
    assert isinstance(root_docs[0], MyDoc)
    assert isinstance(docs[0], SimpleDoc)
    assert len(scores) == 4
    assert sum(score == 1.0 for score in scores) == 2

    for root_doc, doc, score in zip(root_docs, docs, scores):
        assert root_doc.id == f'{doc.id.split("-")[1]}'

        if score == 1.0:
            assert np.allclose(doc.simple_tens, np.ones(10))
        else:
            assert np.allclose(doc.simple_tens, np.ones(10) * 2)


def test_find_subindex_subsublevel(index):
    # sub sub level
    query = np.ones((10,))
    root_docs, docs, scores = index.find_subindex(
        query, subindex='list_docs__docs', search_field='simple_tens', limit=2
    )
    assert len(docs) == 2
    assert isinstance(root_docs[0], MyDoc)
    assert isinstance(docs[0], SimpleDoc)
    for root_doc, doc, score in zip(root_docs, docs, scores):
        assert np.allclose(doc.simple_tens, np.ones(10))
        assert root_doc.id == f'{doc.id.split("-")[2]}'
        assert score == 1.0


def test_subindex_filter(index):
    query = {"simple_doc__simple_text": {"$eq": "hello 1"}}
    docs = index.filter_subindex(query, subindex='list_docs', limit=4)
    assert len(docs) == 2
    assert isinstance(docs[0], ListDoc)
    for doc in docs:
        assert doc.id.split('-')[-1] == '1'

    query = {"simple_text": {"$eq": "hello 0"}}
    docs = index.filter_subindex(query, subindex='list_docs__docs', limit=5)
    assert len(docs) == 4
    assert isinstance(docs[0], SimpleDoc)
    for doc in docs:
        assert doc.id.split('-')[-1] == '0'


def test_subindex_del(index):
    del index['0']
    assert index.num_docs() == 1
    assert index._subindices['docs'].num_docs() == 2
    assert index._subindices['list_docs'].num_docs() == 2
    assert index._subindices['list_docs']._subindices['docs'].num_docs() == 4


def test_subindex_collections(mongo_fixture_env):
    uri, database = mongo_fixture_env
    from typing import Optional

    from pydantic import Field

    from docarray.typing.tensor import AnyTensor

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

    doc_index = MongoAtlasDocumentIndex[MetaCategoryDoc](
        mongo_connection_uri=uri,
        database_name=database,
    )

    assert doc_index._subindices["paths"].index_name == 'metacategorydoc__paths'
    assert doc_index._subindices["paths"]._collection == 'metacategorydoc__paths'
