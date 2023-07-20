import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray


class SchemaDoc(BaseDoc):
    text: str
    price: int
    tensor: NdArray[10]


@pytest.fixture
def docs():
    docs = DocList[SchemaDoc](
        [
            SchemaDoc(text=f'text {i}', price=i, tensor=np.random.rand(10))
            for i in range(9)
        ]
    )
    docs.append(SchemaDoc(text='zd all', price=100, tensor=np.random.rand(10)))
    return docs


@pytest.fixture
def doc_index(docs, tmp_path):
    doc_index = HnswDocumentIndex[SchemaDoc](work_dir=tmp_path)
    doc_index.index(docs)
    return doc_index


def test_query_filter_find_filter(doc_index):
    q = (
        doc_index.build_query()
        .filter(filter_query={'price': {'$lte': 3}})
        .find(query=np.ones(10), search_field='tensor')
        .filter(filter_query={'text': {'$eq': 'text 1'}})
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) == 1
    assert docs[0].price <= 3
    assert docs[0].text == 'text 1'


def test_query_find_filter(doc_index):
    q = (
        doc_index.build_query()
        .find(query=np.ones(10), search_field='tensor')
        .filter(filter_query={'price': {'$gt': 3}}, limit=5)
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) <= 5
    for doc in docs:
        assert doc.price > 3


def test_query_filter_exists_find(doc_index):
    q = (
        doc_index.build_query()
        .filter(filter_query={'text': {'$exists': True}})
        .find(query=np.ones(10), search_field='tensor')
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    # All documents have a 'text' field, so all documents should be returned.
    assert len(docs) == 10


def test_query_filter_not_exists_find(doc_index):
    q = (
        doc_index.build_query()
        .filter(filter_query={'text': {'$exists': False}})
        .find(query=np.ones(10), search_field='tensor')
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    # No documents have missing 'text' field, so no documents should be returned.
    assert len(docs) == 0


def test_query_find_filter_neq(doc_index):
    q = (
        doc_index.build_query()
        .find(query=np.ones(10), search_field='tensor')
        .filter(filter_query={'price': {'$neq': 3}}, limit=5)
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) <= 5
    for doc in docs:
        assert doc.price != 3


def test_query_filter_gte_find(doc_index):
    q = (
        doc_index.build_query()
        .filter(filter_query={'price': {'$gte': 5}})
        .find(query=np.ones(10), search_field='tensor')
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    for doc in docs:
        assert doc.price >= 5


def test_query_filter_lt_find_filter_gt(doc_index):
    q = (
        doc_index.build_query()
        .filter(filter_query={'price': {'$lt': 8}})
        .find(query=np.ones(10), search_field='tensor')
        .filter(filter_query={'price': {'$gt': 2}}, limit=5)
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) <= 5
    for doc in docs:
        assert 2 < doc.price < 8


def test_query_find_filter_and(doc_index):
    q = (
        doc_index.build_query()
        .find(query=np.ones(10), search_field='tensor')
        .filter(
            filter_query={
                '$and': [{'price': {'$gt': 2}}, {'text': {'$neq': 'text 1'}}]
            },
            limit=5,
        )
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) <= 5
    for doc in docs:
        assert doc.price > 2 and doc.text != 'text 1'


def test_query_filter_or_find(doc_index):
    q = (
        doc_index.build_query()
        .filter(
            filter_query={'$or': [{'price': {'$eq': 3}}, {'text': {'$eq': 'text 3'}}]}
        )
        .find(query=np.ones(10), search_field='tensor')
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    for doc in docs:
        assert doc.price == 3 or doc.text == 'text 3'


def test_query_find_filter_not(doc_index):
    q = (
        doc_index.build_query()
        .find(query=np.ones(10), search_field='tensor')
        .filter(filter_query={'$not': {'price': {'$eq': 3}}}, limit=5)
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) <= 5
    for doc in docs:
        assert doc.price != 3
