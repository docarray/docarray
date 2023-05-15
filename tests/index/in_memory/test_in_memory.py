import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index.backends.in_memory import InMemoryExactNNIndex
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
    doc_index = InMemoryExactNNIndex[SchemaDoc]()
    assert doc_index.num_docs() == 0

    doc_index.index(docs)
    assert doc_index.num_docs() == 10


@pytest.fixture
def doc_index(docs):
    doc_index = InMemoryExactNNIndex[SchemaDoc]()
    doc_index.index(docs)
    return doc_index


def test_del_item(docs, doc_index):
    to_remove = [docs[0].id, docs[1].id]
    doc_index._del_items(to_remove)
    assert doc_index.num_docs() == 8


def test_del(docs, doc_index):
    del doc_index[docs[0].id]
    assert doc_index.num_docs() == 9


@pytest.mark.parametrize('space', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
@pytest.mark.parametrize('is_query_doc', [True, False])
def test_find(doc_index, space, is_query_doc):
    class MyDoc(BaseDoc):
        text: str
        price: int
        tensor: NdArray[10] = Field(space=space)

    if is_query_doc:
        query = MyDoc(text='query', price=0, tensor=np.ones(10))
    else:
        query = np.ones(10)

    docs, scores = doc_index.find(query, search_field='tensor', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert doc_index.num_docs() == 10


@pytest.mark.parametrize('space', ['cosine_sim', 'euclidean_dist', 'sqeuclidean_dist'])
@pytest.mark.parametrize('is_query_doc', [True, False])
def test_find_batched(doc_index, space, is_query_doc):
    class MyDoc(BaseDoc):
        text: str
        price: int
        tensor: NdArray[10] = Field(space=space)

    if is_query_doc:
        query = DocList[MyDoc](
            [
                MyDoc(text='query 0', price=0, tensor=np.zeros(10)),
                MyDoc(text='query 1', price=1, tensor=np.ones(10)),
            ]
        )
    else:
        query = np.ones((2, 10))

    docs, scores = doc_index.find_batched(query, search_field='tensor', limit=5)

    assert len(docs) == 2
    for result in docs:
        assert len(result) == 5
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


def test_save_and_load(doc_index, tmpdir):
    initial_num_docs = doc_index.num_docs()

    binary_file = str(tmpdir / 'docs.bin')
    doc_index.save_binary(binary_file)

    new_doc_index = InMemoryExactNNIndex[SchemaDoc](index_file_path=binary_file)

    docs, scores = new_doc_index.find(np.ones(10), search_field='tensor', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert new_doc_index.num_docs() == initial_num_docs
