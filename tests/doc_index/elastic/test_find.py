import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.elastic_doc_index import ElasticDocumentIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.doc_index]


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dims=1000)


class FlatDoc(BaseDocument):
    tens_one: NdArray = Field(dims=10)
    tens_two: NdArray = Field(dims=50)


class NestedDoc(BaseDocument):
    d: SimpleDoc


class DeepNestedDoc(BaseDocument):
    d: NestedDoc


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_simple_schema(similarity):
    class SimpleSchema(BaseDocument):
        tens: NdArray[10] = Field(similarity=similarity)

    store = ElasticDocumentIndex[SimpleSchema]()

    index_docs = []
    for _ in range(10):
        vec = np.random.rand(10)
        if similarity == 'dot_product':
            vec = vec / np.linalg.norm(vec)
        index_docs.append(SimpleDoc(tens=vec))
    store.index(index_docs)

    query = index_docs[-1]
    docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_flat_schema(similarity):
    class FlatSchema(BaseDocument):
        tens_one: NdArray = Field(dims=10, similarity=similarity)
        tens_two: NdArray = Field(dims=50, similarity=similarity)

    store = ElasticDocumentIndex[FlatSchema]()

    index_docs = []
    for _ in range(10):
        vec_one = np.random.rand(10)
        vec_two = np.random.rand(50)
        if similarity == 'dot_product':
            vec_one = vec_one / np.linalg.norm(vec_one)
            vec_two = vec_two / np.linalg.norm(vec_two)
        index_docs.append(FlatDoc(tens_one=vec_one, tens_two=vec_two))

    store.index(index_docs)

    query = index_docs[-1]

    # find on tens_one
    docs, scores = store.find(query, search_field='tens_one', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens_one, index_docs[-1].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-1].tens_two)

    # find on tens_two
    docs, scores = store.find(query, search_field='tens_two', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens_one, index_docs[-1].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-1].tens_two)


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_nested_schema(similarity):
    class SimpleDoc(BaseDocument):
        tens: NdArray[10] = Field(similarity=similarity)

    class NestedDoc(BaseDocument):
        d: SimpleDoc
        tens: NdArray[10] = Field(similarity=similarity)

    class DeepNestedDoc(BaseDocument):
        d: NestedDoc
        tens: NdArray = Field(similarity=similarity, dims=10)

    store = ElasticDocumentIndex[DeepNestedDoc]()

    index_docs = []
    for _ in range(10):
        vec_simple = np.random.rand(10)
        vec_nested = np.random.rand(10)
        vec_deep = np.random.rand(10)
        if similarity == 'dot_product':
            vec_simple = vec_simple / np.linalg.norm(vec_simple)
            vec_nested = vec_nested / np.linalg.norm(vec_nested)
            vec_deep = vec_deep / np.linalg.norm(vec_deep)
        index_docs.append(
            DeepNestedDoc(
                d=NestedDoc(d=SimpleDoc(tens=vec_simple), tens=vec_nested),
                tens=vec_deep,
            )
        )

    store.index(index_docs)

    query = index_docs[-1]

    # find on root level
    docs, scores = store.find(query, search_field='tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)

    # find on first nesting level
    docs, scores = store.find(query, search_field='d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].d.tens, index_docs[-1].d.tens)

    # find on second nesting level
    docs, scores = store.find(query, search_field='d__d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].d.d.tens, index_docs[-1].d.d.tens)


def test_find_batched():
    store = ElasticDocumentIndex[SimpleDoc]()

    index_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(10)]
    store.index(index_docs)

    queries = index_docs[-2:]
    docs_batched, scores_batched = store.find_batched(
        queries, search_field='tens', limit=5
    )

    for docs, scores, query in zip(docs_batched, scores_batched, queries):
        assert len(docs) == 5
        assert len(scores) == 5
        assert docs[0].id == query.id
        assert np.allclose(docs[0].tens, query.tens)


def test_filter():
    import itertools

    class MyDoc(BaseDocument):
        A: bool
        B: int
        C: float

    store = ElasticDocumentIndex[MyDoc]()

    A_list = [True, False]
    B_list = [1, 2]
    C_list = [1.5, 2.5]

    # cross product of all possible combinations
    combinations = itertools.product(A_list, B_list, C_list)
    index_docs = [MyDoc(A=A, B=B, C=C) for A, B, C in combinations]
    store.index(index_docs)

    filter_query = {'term': {'A': True}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.A

    filter_query = {'term': {'B': 1}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.B == 1

    filter_query = {'term': {'C': 1.5}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.C == 1.5


def test_text_search():
    class MyDoc(BaseDocument):
        text: str

    store = ElasticDocumentIndex[MyDoc]()
    index_docs = [
        MyDoc(text='hello world'),
        MyDoc(text='never gonna give you up'),
        MyDoc(text='we are the world'),
    ]
    store.index(index_docs)

    query = 'world'
    docs, scores = store.text_search(query, search_field='text')

    assert len(docs) == 2
    assert len(scores) == 2
    assert docs[0].text.index(query) >= 0
    assert docs[1].text.index(query) >= 0

    queries = ['world', 'never']
    docs, scores = store.text_search_batched(queries, search_field='text')
    print(scores)
    for query, da, score in zip(queries, docs, scores):
        print(score)
        assert len(da) > 0
        assert len(score) > 0
        for doc in da:
            assert doc.text.index(query) >= 0
