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


def test_build_query_eq():
    param_values = []
    query = {'text': {'$eq': 'text 1'}}
    assert HnswDocumentIndex._build_filter_query(query, param_values) == 'text = ?'
    assert param_values == ['text 1']


def test_build_query_lt():
    param_values = []
    query = {'price': {'$lt': 500}}
    assert HnswDocumentIndex._build_filter_query(query, param_values) == 'price < ?'
    assert param_values == [500]


def test_build_query_and():
    param_values = []
    query = {'$and': [{'text': {'$eq': 'text 1'}}, {'price': {'$lt': 500}}]}
    assert (
        HnswDocumentIndex._build_filter_query(query, param_values)
        == '(text = ? AND price < ?)'
    )
    assert param_values == ['text 1', 500]


def test_build_query_invalid_operator():
    param_values = []
    query = {'price': {'$invalid': 500}}
    with pytest.raises(ValueError, match=r"Invalid operator \$invalid"):
        HnswDocumentIndex._build_filter_query(query, param_values)


def test_build_query_invalid_query():
    param_values = []
    query = {'price': 500}
    with pytest.raises(ValueError, match=r"Invalid condition for field price"):
        HnswDocumentIndex._build_filter_query(query, param_values)


def test_filter_eq(doc_index, docs):
    filter_result = doc_index.filter({'text': {'$eq': 'text 1'}})
    assert len(filter_result) == 1
    assert filter_result[0].text == 'text 1'
    assert filter_result[0].text == docs[1].text
    assert filter_result[0].price == docs[1].price
    assert filter_result[0].id == docs[1].id
    assert np.allclose(filter_result[0].tensor, docs[1].tensor)


def test_filter_neq(doc_index):
    docs = doc_index.filter({'text': {'$neq': 'text 1'}})
    assert len(docs) == 9
    assert all(doc.text != 'text 1' for doc in docs)


def test_filter_lt(doc_index):
    docs = doc_index.filter({'price': {'$lt': 3}})
    assert len(docs) == 3
    assert all(doc.price < 3 for doc in docs)


def test_filter_lte(doc_index):
    docs = doc_index.filter({'price': {'$lte': 2}})
    assert len(docs) == 3
    assert all(doc.price <= 2 for doc in docs)


def test_filter_gt(doc_index):
    docs = doc_index.filter({'price': {'$gt': 5}})
    assert len(docs) == 4
    assert all(doc.price > 5 for doc in docs)


def test_filter_gte(doc_index):
    docs = doc_index.filter({'price': {'$gte': 6}})
    assert len(docs) == 4
    assert all(doc.price >= 6 for doc in docs)


def test_filter_exists(doc_index):
    docs = doc_index.filter({'price': {'$exists': True}})
    assert len(docs) == 10
    assert all(hasattr(doc, 'price') for doc in docs)


def test_filter_or(doc_index):
    docs = doc_index.filter(
        {
            '$or': [
                {'text': {'$eq': 'text 1'}},
                {'price': {'$eq': 2}},
            ]
        }
    )
    assert len(docs) == 2
    assert any(doc.text == 'text 1' or doc.price == 2 for doc in docs)


def test_filter_and(doc_index):
    docs = doc_index.filter(
        {
            '$and': [
                {'text': {'$eq': 'text 1'}},
                {'price': {'$eq': 1}},
            ]
        }
    )
    assert len(docs) == 1
    assert any(doc.text == 'text 1' and doc.price == 1 for doc in docs)


def test_filter_not(doc_index):
    docs = doc_index.filter({'$not': {'text': {'$eq': 'text 1'}}})
    assert len(docs) == 9
    assert all(doc.text != 'text 1' for doc in docs)


def test_filter_not_and(doc_index):
    docs = doc_index.filter(
        {
            '$not': {
                '$and': [
                    {'text': {'$eq': 'text 1'}},
                    {'price': {'$eq': 1}},
                ]
            }
        }
    )
    assert len(docs) == 9
    assert all(not (doc.text == 'text 1' and doc.price == 1) for doc in docs)
