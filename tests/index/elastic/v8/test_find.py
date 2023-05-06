import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.elastic.fixture import (  # noqa: F401
    FlatDoc,
    SimpleDoc,
    start_storage_v8,
    tmp_index_name,
)

pytestmark = [pytest.mark.slow, pytest.mark.index, pytest.mark.elasticv8]


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_simple_schema(similarity, tmp_index_name):  # noqa: F811
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(similarity=similarity)

    index = ElasticDocIndex[SimpleSchema](index_name=tmp_index_name)

    index_docs = []
    for _ in range(10):
        vec = np.random.rand(10)
        if similarity == 'dot_product':
            vec = vec / np.linalg.norm(vec)
        index_docs.append(SimpleDoc(tens=vec))
    index.index(index_docs)

    query = index_docs[-1]
    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_flat_schema(similarity, tmp_index_name):  # noqa: F811
    class FlatSchema(BaseDoc):
        tens_one: NdArray = Field(dims=10, similarity=similarity)
        tens_two: NdArray = Field(dims=50, similarity=similarity)

    index = ElasticDocIndex[FlatSchema](index_name=tmp_index_name)

    index_docs = []
    for _ in range(10):
        vec_one = np.random.rand(10)
        vec_two = np.random.rand(50)
        if similarity == 'dot_product':
            vec_one = vec_one / np.linalg.norm(vec_one)
            vec_two = vec_two / np.linalg.norm(vec_two)
        index_docs.append(FlatDoc(tens_one=vec_one, tens_two=vec_two))

    index.index(index_docs)

    query = index_docs[-1]

    # find on tens_one
    docs, scores = index.find(query, search_field='tens_one', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens_one, index_docs[-1].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-1].tens_two)

    # find on tens_two
    docs, scores = index.find(query, search_field='tens_two', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens_one, index_docs[-1].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-1].tens_two)


@pytest.mark.parametrize('similarity', ['cosine', 'l2_norm', 'dot_product'])
def test_find_nested_schema(similarity, tmp_index_name):  # noqa: F811
    class SimpleDoc(BaseDoc):
        tens: NdArray[10] = Field(similarity=similarity)

    class NestedDoc(BaseDoc):
        d: SimpleDoc
        tens: NdArray[10] = Field(similarity=similarity)

    class DeepNestedDoc(BaseDoc):
        d: NestedDoc
        tens: NdArray = Field(similarity=similarity, dims=10)

    index = ElasticDocIndex[DeepNestedDoc](index_name=tmp_index_name)

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

    index.index(index_docs)

    query = index_docs[-1]

    # find on root level
    docs, scores = index.find(query, search_field='tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)

    # find on first nesting level
    docs, scores = index.find(query, search_field='d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].d.tens, index_docs[-1].d.tens)

    # find on second nesting level
    docs, scores = index.find(query, search_field='d__d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].d.d.tens, index_docs[-1].d.d.tens)


def test_find_torch():
    class TorchDoc(BaseDoc):
        tens: TorchTensor[10]

    index = ElasticDocIndex[TorchDoc]()

    # A dense_vector field stores dense vectors of float values.
    index_docs = [
        TorchDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = index_docs[-1]
    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TorchTensor)

    assert docs[0].id == index_docs[-1].id
    assert torch.allclose(docs[0].tens, index_docs[-1].tens)


def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10]

    index = ElasticDocIndex[TfDoc]()

    index_docs = [
        TfDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = index_docs[-1]
    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(
        docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )


def test_find_batched(tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)

    index_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(10)]
    index.index(index_docs)

    queries = index_docs[-2:]
    docs_batched, scores_batched = index.find_batched(
        queries, search_field='tens', limit=5
    )

    for docs, scores, query in zip(docs_batched, scores_batched, queries):
        assert len(docs) == 5
        assert len(scores) == 5
        assert docs[0].id == query.id
        assert np.allclose(docs[0].tens, query.tens)


def test_filter():
    class MyDoc(BaseDoc):
        A: bool
        B: int
        C: float

    index = ElasticDocIndex[MyDoc]()

    index_docs = [MyDoc(id=f'{i}', A=(i % 2 == 0), B=i, C=i + 0.5) for i in range(10)]
    index.index(index_docs)

    filter_query = {'term': {'A': True}}
    docs = index.filter(filter_query)
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
    docs = index.filter(filter_query)
    assert [doc.id for doc in docs] == ['3', '4']


def test_text_search():
    class MyDoc(BaseDoc):
        text: str

    index = ElasticDocIndex[MyDoc]()
    index_docs = [
        MyDoc(text='hello world'),
        MyDoc(text='never gonna give you up'),
        MyDoc(text='we are the world'),
    ]
    index.index(index_docs)

    query = 'world'
    docs, scores = index.text_search(query, search_field='text')

    assert len(docs) == 2
    assert len(scores) == 2
    assert docs[0].text.index(query) >= 0
    assert docs[1].text.index(query) >= 0

    queries = ['world', 'never']
    docs, scores = index.text_search_batched(queries, search_field='text')
    for query, da, score in zip(queries, docs, scores):
        assert len(da) > 0
        assert len(score) > 0
        for doc in da:
            assert doc.text.index(query) >= 0


def test_query_builder():
    class MyDoc(BaseDoc):
        tens: NdArray[10] = Field(similarity='l2_norm')
        num: int
        text: str

    index = ElasticDocIndex[MyDoc]()
    index_docs = [
        MyDoc(
            id=f'{i}', tens=np.ones(10) * i, num=int(i / 2), text=f'text {int(i / 2)}'
        )
        for i in range(10)
    ]
    index.index(index_docs)

    # build_query
    q = index.build_query()
    assert isinstance(q, index.QueryBuilder)

    # filter
    q = index.build_query().filter({'term': {'num': 0}}).build()
    docs, _ = index.execute_query(q)
    assert [doc['id'] for doc in docs] == ['0', '1']

    # find
    q = index.build_query().find(index_docs[-1], search_field='tens', limit=3).build()
    docs, _ = index.execute_query(q)
    assert [doc['id'] for doc in docs] == ['9', '8', '7']

    # text_search
    q = index.build_query().text_search('0', search_field='text').build()
    docs, _ = index.execute_query(q)
    assert [doc['id'] for doc in docs] == ['0', '1']

    # combination
    q = (
        index.build_query()
        .filter({'range': {'num': {'lte': 3}}})
        .find(index_docs[-1], search_field='tens')
        .text_search('0', search_field='text')
        .build()
    )
    docs, _ = index.execute_query(q)
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

    docs, _ = index.execute_query(query)
    assert [doc['id'] for doc in docs] == ['7', '6', '5', '4']


def test_index_name():
    class MyDoc(BaseDoc):
        expected_attendees: dict = Field(col_type='integer_range')
        time_frame: dict = Field(col_type='date_range', format='yyyy-MM-dd')

    index = ElasticDocIndex[MyDoc]()
    assert index.index_name == MyDoc.__name__.lower()
