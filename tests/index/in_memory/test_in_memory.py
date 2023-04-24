import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.index.backends.in_memory import InMemoryDocIndex
from docarray.typing import NdArray


class SchemaDoc(BaseDoc):
    text: str
    price: int
    tensor: NdArray[10]


@pytest.fixture
def docs():
    docs = DocList[SchemaDoc](
        [
            SchemaDoc(text=f'hello {i}', price=i, tensor=np.array([i] * 10))
            for i in range(9)
        ]
    )
    docs.append(SchemaDoc(text='good bye', price=100, tensor=np.array([100.0] * 10)))
    return docs


def test_indexing(docs):
    doc_index = InMemoryDocIndex[SchemaDoc]()
    assert doc_index.num_docs() == 0

    doc_index.index(docs)
    assert doc_index.num_docs() == 10


@pytest.fixture
def doc_index(docs):
    doc_index = InMemoryDocIndex[SchemaDoc]()
    doc_index.index(docs)
    return doc_index


def test_del_item(docs, doc_index):
    to_remove = [docs[0].id, docs[1].id]
    doc_index._del_items(to_remove)
    assert doc_index.num_docs() == 8


def test_find(doc_index):
    query = SchemaDoc(text='query', price=0, tensor=np.ones(10))
    docs, scores = doc_index.find(query, search_field='tensor', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert doc_index.num_docs() == 10


def test_concatenated_queries(doc_index):
    query = SchemaDoc(text='query', price=0, tensor=np.ones(10))

    q = (
        doc_index.build_query()
        .find(query=query, search_field='tensor', limit=5)
        .filter(filter_query={'price': {'$neq': 5}})
        .build()
    )

    docs, scores = doc_index.execute_query(q)

    assert len(docs) == 4
