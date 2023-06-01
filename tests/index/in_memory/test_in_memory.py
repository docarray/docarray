from typing import Optional

import numpy as np
import pytest
from pydantic import Field
from torch import rand

from docarray import BaseDoc, DocList
from docarray.index.backends.in_memory import InMemoryExactNNIndex
from docarray.typing import NdArray, TorchTensor


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

    empty_index = InMemoryExactNNIndex[MyDoc]()
    docs, scores = empty_index.find(query, search_field='tensor', limit=5)
    assert len(docs) == 0
    assert len(scores) == 0


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

    empty_index = InMemoryExactNNIndex[MyDoc]()
    docs, scores = empty_index.find_batched(query, search_field='tensor', limit=5)
    assert len(docs) == 0
    assert len(scores) == 0


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
    doc_index.persist(binary_file)

    new_doc_index = InMemoryExactNNIndex[SchemaDoc](index_file_path=binary_file)

    docs, scores = new_doc_index.find(np.ones(10), search_field='tensor', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert new_doc_index.num_docs() == initial_num_docs

    newer_doc_index = InMemoryExactNNIndex[SchemaDoc](
        index_file_path='some_nonexistent_file.bin'
    )

    assert newer_doc_index.num_docs() == 0


def test_index_with_None_embedding():
    class DocTest(BaseDoc):
        index: int
        embedding: Optional[NdArray[4]]

    # Some of the documents have the embedding field set to None
    dl = DocList[DocTest](
        [
            DocTest(index=i, embedding=np.random.rand(4) if i % 2 else None)
            for i in range(100)
        ]
    )

    index = InMemoryExactNNIndex[DocTest](dl)
    res = index.find(np.random.rand(4), search_field="embedding", limit=70)
    assert len(res.documents) == 50
    for doc in res.documents:
        assert doc.index % 2 != 0


def test_index_avoid_stack_embedding():
    class MyDoc(BaseDoc):
        embedding1: TorchTensor
        embedding2: TorchTensor
        embedding3: TorchTensor

    data = DocList[MyDoc](
        [
            MyDoc(
                embedding1=rand(128),
                embedding2=rand(128),
                embedding3=rand(128),
            )
            for _ in range(10)
        ]
    )

    db = InMemoryExactNNIndex[MyDoc](data)

    query = MyDoc(
        embedding1=rand(128),
        embedding2=rand(128),
        embedding3=rand(128),
    )

    for i in range(3):
        db.find(query, search_field=f"embedding{i + 1}")
        assert len(db._embedding_map) == i + 1

    data_copy = data.copy()

    for i in range(9):
        db._del_items(data_copy[i].id)
        assert db._embedding_map["embedding1"][0].shape[0] == db.num_docs()

    db._del_items(data_copy[9].id)  # Delete the last element
    assert len(db._embedding_map) == 0


def test_index_find_speedup():
    class MyDocument(BaseDoc):
        embedding: TorchTensor
        embedding2: TorchTensor
        embedding3: TorchTensor

    def generate_doc_list(num_docs: int, dims: int) -> DocList[MyDocument]:
        return DocList[MyDocument](
            [
                MyDocument(
                    embedding=rand(dims),
                    embedding2=rand(dims),
                    embedding3=rand(dims),
                )
                for _ in range(num_docs)
            ]
        )

    def create_inmemory_index(
        data_list: DocList[MyDocument],
    ) -> InMemoryExactNNIndex[MyDocument]:
        return InMemoryExactNNIndex[MyDocument](data_list)

    def find_similar_docs(
        index: InMemoryExactNNIndex[MyDocument],
        queries: DocList[MyDocument],
        search_field: str = 'embedding',
        limit: int = 5,
    ) -> tuple:
        return index.find_batched(queries, search_field=search_field, limit=limit)

    # Generating document lists
    num_docs, num_queries, dims = 2000, 1000, 128
    data_list = generate_doc_list(num_docs, dims)
    queries = generate_doc_list(num_queries, dims)

    # Creating index
    db = create_inmemory_index(data_list)

    # Finding similar documents
    for _ in range(5):
        matches, scores = find_similar_docs(db, queries, 'embedding', 5)
        assert len(matches) == num_queries
        assert len(matches[0]) == 5
