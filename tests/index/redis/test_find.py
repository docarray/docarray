from typing import Optional

import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.redis.fixtures import start_redis  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]

N_DIM = 10


def get_simple_schema(**kwargs):
    class SimpleSchema(BaseDoc):
        tens: NdArray[N_DIM] = Field(**kwargs)

    return SimpleSchema


class TorchDoc(BaseDoc):
    tens: TorchTensor[N_DIM]


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_simple_schema(space):
    schema = get_simple_schema(space=space)
    db = RedisDocumentIndex[schema](host='localhost')

    index_docs = [schema(tens=np.random.rand(N_DIM)) for _ in range(10)]
    index_docs.append(schema(tens=np.ones(N_DIM)))

    db.index(index_docs)

    query = schema(tens=np.ones(N_DIM))

    docs, scores = db.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)


def test_find_empty_index():
    schema = get_simple_schema()
    empty_index = RedisDocumentIndex[schema](host='localhost')
    query = schema(tens=np.random.rand(N_DIM))

    docs, scores = empty_index.find(query, search_field='tens', limit=5)
    assert len(docs) == 0
    assert len(scores) == 0


def test_find_limit_larger_than_index():
    schema = get_simple_schema()
    db = RedisDocumentIndex[schema](host='localhost')
    query = schema(tens=np.ones(N_DIM))
    index_docs = [schema(tens=np.zeros(N_DIM)) for _ in range(10)]
    db.index(index_docs)
    docs, scores = db.find(query, search_field='tens', limit=20)
    assert len(docs) == 10
    assert len(scores) == 10


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_torch(space):
    db = RedisDocumentIndex[TorchDoc](host='localhost')
    index_docs = [TorchDoc(tens=np.random.rand(N_DIM)) for _ in range(10)]
    index_docs.append(TorchDoc(tens=np.ones(N_DIM, dtype=np.float32)))
    db.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = TorchDoc(tens=np.ones(N_DIM, dtype=np.float32))

    result_docs, scores = db.find(query, search_field='tens', limit=5)

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TorchTensor)
    assert result_docs[0].id == index_docs[-1].id
    assert torch.allclose(result_docs[0].tens, index_docs[-1].tens)


@pytest.mark.tensorflow
@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_tensorflow(space):
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10]

    db = RedisDocumentIndex[TorchDoc](host='localhost')

    index_docs = [TfDoc(tens=np.random.rand(N_DIM)) for _ in range(10)]
    index_docs.append(TfDoc(tens=np.ones(10)))
    db.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = TfDoc(tens=np.ones(10))

    result_docs, scores = db.find(query, search_field='tens', limit=5)

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TensorFlowTensor)
    assert result_docs[0].id == index_docs[-1].id
    assert np.allclose(
        result_docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_flat_schema(space):
    class FlatSchema(BaseDoc):
        tens_one: NdArray = Field(dim=N_DIM, space=space)
        tens_two: NdArray = Field(dim=50, space=space)

    index = RedisDocumentIndex[FlatSchema](host='localhost')

    index_docs = [
        FlatSchema(tens_one=np.random.rand(N_DIM), tens_two=np.random.rand(50))
        for _ in range(10)
    ]
    index_docs.append(FlatSchema(tens_one=np.zeros(N_DIM), tens_two=np.ones(50)))
    index_docs.append(FlatSchema(tens_one=np.ones(N_DIM), tens_two=np.zeros(50)))
    index.index(index_docs)

    query = FlatSchema(tens_one=np.ones(N_DIM), tens_two=np.ones(50))

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
    assert docs[0].id == index_docs[-2].id
    assert np.allclose(docs[0].tens_one, index_docs[-2].tens_one)
    assert np.allclose(docs[0].tens_two, index_docs[-2].tens_two)


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_nested_schema(space):
    class SimpleDoc(BaseDoc):
        tens: NdArray[N_DIM] = Field(space=space)

    class NestedDoc(BaseDoc):
        d: SimpleDoc
        tens: NdArray[N_DIM] = Field(space=space)

    class DeepNestedDoc(BaseDoc):
        d: NestedDoc
        tens: NdArray = Field(space=space, dim=N_DIM)

    index = RedisDocumentIndex[DeepNestedDoc](host='localhost')

    index_docs = [
        DeepNestedDoc(
            d=NestedDoc(
                d=SimpleDoc(tens=np.random.rand(N_DIM)), tens=np.random.rand(N_DIM)
            ),
            tens=np.random.rand(N_DIM),
        )
        for _ in range(10)
    ]
    index_docs.append(
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.ones(N_DIM)), tens=np.zeros(N_DIM)),
            tens=np.zeros(N_DIM),
        )
    )
    index_docs.append(
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.zeros(N_DIM)), tens=np.ones(N_DIM)),
            tens=np.zeros(N_DIM),
        )
    )
    index_docs.append(
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.zeros(N_DIM)), tens=np.zeros(N_DIM)),
            tens=np.ones(N_DIM),
        )
    )
    index.index(index_docs)

    query = DeepNestedDoc(
        d=NestedDoc(d=SimpleDoc(tens=np.ones(N_DIM)), tens=np.ones(N_DIM)),
        tens=np.ones(N_DIM),
    )

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
    assert docs[0].id == index_docs[-2].id
    assert np.allclose(docs[0].d.tens, index_docs[-2].d.tens)

    # find on second nesting level
    docs, scores = index.find(query, search_field='d__d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-3].id
    assert np.allclose(docs[0].d.d.tens, index_docs[-3].d.d.tens)


def test_simple_usage():
    class MyDoc(BaseDoc):
        text: str
        embedding: NdArray[128]

    docs = [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    queries = docs[0:3]
    index = RedisDocumentIndex[MyDoc](host='localhost')
    index.index(docs=DocList[MyDoc](docs))
    resp = index.find_batched(queries=queries, search_field='embedding', limit=10)
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 10
        assert q.id == matches[0].id


def test_query_builder():
    class SimpleSchema(BaseDoc):
        tensor: NdArray[N_DIM] = Field(space='cosine')
        price: int

    db = RedisDocumentIndex[SimpleSchema](host='localhost')

    index_docs = [
        SimpleSchema(tensor=np.array([i + 1] * 10), price=i + 1) for i in range(10)
    ]
    db.index(index_docs)

    q = (
        db.build_query()
        .find(query=np.ones(N_DIM), search_field='tensor', limit=5)
        .filter(filter_query='@price:[-inf 3]')
        .build()
    )

    docs, scores = db.execute_query(q)

    assert len(docs) == 3
    for doc in docs:
        assert doc.price <= 3


def test_text_search():
    class SimpleSchema(BaseDoc):
        description: str
        some_field: Optional[int]

    texts_to_index = [
        "Text processing with Python is a valuable skill for data analysis.",
        "Gardening tips for a beautiful backyard oasis.",
        "Explore the wonders of deep-sea diving in tropical locations.",
        "The history and art of classical music compositions.",
        "An introduction to the world of gourmet cooking.",
    ]

    query_string = "Python and text processing"

    docs = [SimpleSchema(description=text) for text in texts_to_index]

    db = RedisDocumentIndex[SimpleSchema](host='localhost')
    db.index(docs)

    docs, _ = db.text_search(query=query_string, search_field='description')

    assert docs[0].description == texts_to_index[0]


def test_filter():
    class SimpleSchema(BaseDoc):
        description: str
        price: int

    doc1 = SimpleSchema(description='Python book', price=50)
    doc2 = SimpleSchema(description='Python book by some author', price=60)
    doc3 = SimpleSchema(description='Random book', price=40)
    docs = [doc1, doc2, doc3]

    db = RedisDocumentIndex[SimpleSchema](host='localhost')
    db.index(docs)

    # filter on price < 45
    docs = db.filter(filter_query='@price:[-inf 45]')
    assert len(docs) == 1
    assert docs[0].price == 40

    # filter on price >= 50
    docs = db.filter(filter_query='@price:[50 inf]')
    assert len(docs) == 2
    for doc in docs:
        assert doc.price >= 50

    # get documents with the phrase "python book" in the description
    docs = db.filter(filter_query='@description:"python book"')
    assert len(docs) == 2
    for doc in docs:
        assert 'python book' in doc.description.lower()

    # get documents with the word "book" in the description that have price <= 45
    docs = db.filter(filter_query='@description:"book" @price:[-inf 45]')
    assert len(docs) == 1
    assert docs[0].description == 'Random book' and docs[0].price == 40
