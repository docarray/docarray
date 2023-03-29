import numpy as np
import torch
from pydantic import Field

from docarray import BaseDocument
from docarray.index import ElasticV7DocIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.elastic.fixture import start_storage_v7  # noqa: F401
from tests.index.elastic.fixture import FlatDoc, SimpleDoc


def test_find_simple_schema():
    class SimpleSchema(BaseDocument):
        tens: NdArray[10]

    store = ElasticV7DocIndex[SimpleSchema]()

    index_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(10)]
    store.index(index_docs)

    query = index_docs[-1]
    docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)


def test_find_flat_schema():
    class FlatSchema(BaseDocument):
        tens_one: NdArray = Field(dims=10)
        tens_two: NdArray = Field(dims=50)

    store = ElasticV7DocIndex[FlatSchema]()

    index_docs = [
        FlatDoc(tens_one=np.random.rand(10), tens_two=np.random.rand(50))
        for _ in range(10)
    ]
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


def test_find_nested_schema():
    class SimpleDoc(BaseDocument):
        tens: NdArray[10]

    class NestedDoc(BaseDocument):
        d: SimpleDoc
        tens: NdArray[10]

    class DeepNestedDoc(BaseDocument):
        d: NestedDoc
        tens: NdArray = Field(dims=10)

    store = ElasticV7DocIndex[DeepNestedDoc]()

    index_docs = [
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.random.rand(10)), tens=np.random.rand(10)),
            tens=np.random.rand(10),
        )
        for _ in range(10)
    ]
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


def test_find_torch():
    class TorchDoc(BaseDocument):
        tens: TorchTensor[10]

    store = ElasticV7DocIndex[TorchDoc]()

    # A dense_vector field stores dense vectors of float values.
    index_docs = [
        TorchDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    store.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = index_docs[-1]
    docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TorchTensor)

    assert docs[0].id == index_docs[-1].id
    assert torch.allclose(docs[0].tens, index_docs[-1].tens)


def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDocument):
        tens: TensorFlowTensor[10]

    store = ElasticV7DocIndex[TfDoc]()

    index_docs = [
        TfDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    store.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = index_docs[-1]
    docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(
        docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )


def test_find_batched():
    store = ElasticV7DocIndex[SimpleDoc]()

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
    class MyDoc(BaseDocument):
        A: bool
        B: int
        C: float

    store = ElasticV7DocIndex[MyDoc]()

    index_docs = [MyDoc(id=f'{i}', A=(i % 2 == 0), B=i, C=i + 0.5) for i in range(10)]
    store.index(index_docs)

    filter_query = {'term': {'A': True}}
    docs = store.filter(filter_query)
    assert len(docs) > 0
    for doc in docs:
        assert doc.A

    filter_query = {
        'bool': {
            'filter': [
                {'terms': {'B': [3, 4, 7, 8]}},
                {'range': {'C': {'gte': 3, 'lte': 5}}},
            ]
        }
    }
    docs = store.filter(filter_query)
    assert [doc.id for doc in docs] == ['3', '4']


def test_text_search():
    class MyDoc(BaseDocument):
        text: str

    store = ElasticV7DocIndex[MyDoc]()
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
        tens: NdArray[10]
        num: int
        text: str

    store = ElasticV7DocIndex[MyDoc]()
    index_docs = [
        MyDoc(
            id=f'{i}', tens=np.random.rand(10), num=int(i / 2), text=f'text {int(i/2)}'
        )
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
    docs, scores = store.execute_query(q)
    assert len(docs) == 3
    assert len(scores) == 3
    assert docs[0]['id'] == index_docs[-1].id
    assert np.allclose(docs[0]['tens'], index_docs[-1].tens)

    # text search
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
    assert sorted([doc['id'] for doc in docs]) == ['0', '1']

    # direct
    index_docs = [
        MyDoc(id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'text {int(i/2)}')
        for i in range(10)
    ]
    store.index(index_docs)

    query = {
        'query': {
            'script_score': {
                'query': {
                    'bool': {
                        'filter': [
                            {'range': {'num': {'gte': 2}}},
                            {'range': {'num': {'lte': 3}}},
                        ],
                    },
                },
                'script': {
                    'source': '1 / (1 + l2norm(params.query_vector, \'tens\'))',
                    'params': {'query_vector': index_docs[-1].tens},
                },
            }
        }
    }

    docs, _ = store.execute_query(query)
    assert [doc['id'] for doc in docs] == ['7', '6', '5', '4']
