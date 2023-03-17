import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.elasticv8_doc_index import ElasticDocumentV8Index
from docarray.typing import NdArray
from tests.doc_index.elastic.fixture import start_storage_v8  # noqa: F401
from tests.doc_index.elastic.fixture import FlatDoc, SimpleDoc


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_simple_schema(similarity):
    class SimpleSchema(BaseDocument):
        tens: NdArray[10] = Field(similarity=similarity)

    store = ElasticDocumentV8Index[SimpleSchema]()

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

    store = ElasticDocumentV8Index[FlatSchema]()

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

    store = ElasticDocumentV8Index[DeepNestedDoc]()

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
    store = ElasticDocumentV8Index[SimpleDoc]()

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

    store = ElasticDocumentV8Index[MyDoc]()

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

    store = ElasticDocumentV8Index[MyDoc]()
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
    for query, da, score in zip(queries, docs, scores):
        assert len(da) > 0
        assert len(score) > 0
        for doc in da:
            assert doc.text.index(query) >= 0


def test_query_builder():
    class MyDoc(BaseDocument):
        tens: NdArray[10] = Field(similarity='l2_norm')
        num: int
        text: str

    store = ElasticDocumentV8Index[MyDoc]()
    index_docs = [
        MyDoc(id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'text {int(i/2)}')
        for i in range(10)
    ]
    store.index(index_docs)

    # build_query
    q = store.build_query()
    assert isinstance(q, store.QueryBuilder)

    # filter
    q = store.build_query().filter({'term': {'num': 0}}).build()
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['0', '1']

    # find
    q = store.build_query().find(index_docs[-1], search_field='tens', limit=3).build()
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['9', '8', '7']

    # text_search
    q = store.build_query().text_search('0', search_field='text').build()
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['0', '1']

    # combination
    q = (
        store.build_query()
        .filter({'range': {'num': {'lte': 3}}})
        .find(index_docs[-1], search_field='tens')
        .text_search('0', search_field='text')
        .build()
    )
    docs, _ = store.execute_query(q)
    assert [doc['id'] for doc in docs] == ['1', '0']

    # direct
    query = {
        'knn': {
            'field': 'tens',
            'query_vector': [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            'k': 10,
            'num_candidates': 10000,
            'filter': {
                'bool': {
                    'filter': [
                        {'range': {'num': {'gte': 2}}},
                        {'range': {'num': {'lte': 3}}},
                    ]
                }
            },
        },
    }

    docs, _ = store.execute_query(query)
    assert [doc['id'] for doc in docs] == ['7', '6', '5', '4']
