import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray, TorchTensor

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDoc):
    d: SimpleDoc


class DeepNestedDoc(BaseDoc):
    d: NestedDoc


class TorchDoc(BaseDoc):
    tens: TorchTensor[10]


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_simple_schema(tmp_path, space):
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(space=space)

    index = HnswDocumentIndex[SimpleSchema](work_dir=str(tmp_path))

    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(SimpleDoc(tens=np.ones(10)))
    index.index(index_docs)

    query = SimpleDoc(tens=np.ones(10))

    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)
    for result in docs[1:]:
        assert np.allclose(result.tens, np.zeros(10))


def test_find_empty_index(tmp_path):
    empty_index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    query = SimpleDoc(tens=np.ones(10))

    docs, scores = empty_index.find(query, search_field='tens', limit=5)
    assert len(docs) == 0
    assert len(scores) == 0


def test_find_limit_larger_than_index(tmp_path):
    index = HnswDocumentIndex[SimpleDoc](work_dir=str(tmp_path))
    query = SimpleDoc(tens=np.ones(10))
    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index.index(index_docs)
    docs, scores = index.find(query, search_field='tens', limit=20)
    assert len(docs) == 10
    assert len(scores) == 10


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_torch(tmp_path, space):
    index = HnswDocumentIndex[TorchDoc](work_dir=str(tmp_path))

    index_docs = [TorchDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(TorchDoc(tens=np.ones(10)))
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = TorchDoc(tens=np.ones(10))

    result_docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TorchTensor)
    assert result_docs[0].id == index_docs[-1].id
    assert torch.allclose(result_docs[0].tens, index_docs[-1].tens)
    for result in result_docs[1:]:
        assert torch.allclose(result.tens, torch.zeros(10, dtype=torch.float64))


@pytest.mark.tensorflow
def test_find_tensorflow(tmp_path):
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10]

    index = HnswDocumentIndex[TfDoc](work_dir=str(tmp_path))

    index_docs = [TfDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(TfDoc(tens=np.ones(10)))
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = TfDoc(tens=np.ones(10))

    result_docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TensorFlowTensor)
    assert result_docs[0].id == index_docs[-1].id
    assert np.allclose(
        result_docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )
    for result in result_docs[1:]:
        assert np.allclose(result.tens.unwrap().numpy(), np.zeros(10))


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_flat_schema(tmp_path, space):
    class FlatSchema(BaseDoc):
        tens_one: NdArray = Field(dim=10, space=space)
        tens_two: NdArray = Field(dim=50, space=space)

    index = HnswDocumentIndex[FlatSchema](work_dir=str(tmp_path))

    index_docs = [
        FlatDoc(tens_one=np.zeros(10), tens_two=np.zeros(50)) for _ in range(10)
    ]
    index_docs.append(FlatDoc(tens_one=np.zeros(10), tens_two=np.ones(50)))
    index_docs.append(FlatDoc(tens_one=np.ones(10), tens_two=np.zeros(50)))
    index.index(index_docs)

    query = FlatDoc(tens_one=np.ones(10), tens_two=np.ones(50))

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
def test_find_nested_schema(tmp_path, space):
    class SimpleDoc(BaseDoc):
        tens: NdArray[10] = Field(space=space)

    class NestedDoc(BaseDoc):
        d: SimpleDoc
        tens: NdArray[10] = Field(space=space)

    class DeepNestedDoc(BaseDoc):
        d: NestedDoc
        tens: NdArray = Field(space=space, dim=10)

    index = HnswDocumentIndex[DeepNestedDoc](work_dir=str(tmp_path))

    index_docs = [
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.zeros(10)), tens=np.zeros(10)),
            tens=np.zeros(10),
        )
        for _ in range(10)
    ]
    index_docs.append(
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.ones(10)), tens=np.zeros(10)),
            tens=np.zeros(10),
        )
    )
    index_docs.append(
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.zeros(10)), tens=np.ones(10)),
            tens=np.zeros(10),
        )
    )
    index_docs.append(
        DeepNestedDoc(
            d=NestedDoc(d=SimpleDoc(tens=np.zeros(10)), tens=np.zeros(10)),
            tens=np.ones(10),
        )
    )
    index.index(index_docs)

    query = DeepNestedDoc(
        d=NestedDoc(d=SimpleDoc(tens=np.ones(10)), tens=np.ones(10)), tens=np.ones(10)
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


def test_simple_usage(tmpdir):
    class MyDoc(BaseDoc):
        text: str
        embedding: NdArray[128]

    docs = [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    queries = docs[0:3]
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir), index_name='index')
    index.index(docs=DocList[MyDoc](docs))
    resp = index.find_batched(queries=queries, search_field='embedding', limit=10)
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 10
        assert q.id == matches[0].id


def test_usage_adapt_max_elements(tmpdir):
    class MyDoc(BaseDoc):
        text: str
        embedding: NdArray[128]

    docs = DocList[MyDoc](
        [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    )
    queries = docs[0:3]
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.configure()  # trying to configure the index but I am not managing to do so.
    index.index(docs=docs)
    resp = index.find_batched(queries=queries, search_field='embedding', limit=10)
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 10
        assert q.id == matches[0].id


def test_usage_adapt_max_elements_after_restore(tmpdir):
    class MyDoc(BaseDoc):
        text: str
        embedding: NdArray[128]

    docs = DocList[MyDoc](
        [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    )
    queries = docs[0:3]
    index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    index.configure()  # trying to configure the index but I am not managing to do so.
    index.index(docs=docs)
    resp = index.find_batched(queries=queries, search_field='embedding', limit=10)
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 10
        assert q.id == matches[0].id

    new_docs = DocList[MyDoc](
        [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    )
    restored_index = HnswDocumentIndex[MyDoc](work_dir=str(tmpdir))
    restored_index.index(docs=new_docs)
    queries = new_docs[0:3]
    resp = restored_index.find_batched(
        queries=queries, search_field='embedding', limit=10
    )
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 10
        assert q.id == matches[0].id


@pytest.mark.parametrize(
    'find_limit, filter_limit, expected_docs', [(10, 3, 3), (5, None, 5)]
)
def test_query_builder_limits(find_limit, filter_limit, expected_docs, tmp_path):
    class SimpleSchema(BaseDoc):
        tensor: NdArray[10] = Field(space='l2')
        price: int

    index = HnswDocumentIndex[SimpleSchema](work_dir=str(tmp_path))

    index_docs = [SimpleSchema(tensor=np.array([i] * 10), price=i) for i in range(10)]
    index.index(index_docs)

    query = SimpleSchema(tensor=np.array([3] * 10), price=3)

    q = (
        index.build_query()
        .find(query=query, search_field='tensor', limit=find_limit)
        .filter(filter_query={'price': {'$lte': 5}}, limit=filter_limit)
        .build()
    )

    docs, scores = index.execute_query(q)

    assert len(docs) == expected_docs


def test_contain(tmp_path):
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(space="cosine")

    index = HnswDocumentIndex[SimpleSchema](work_dir=str(tmp_path))
    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index.index(index_docs)

    for doc in index_docs:
        assert (doc in index) is True

    index_docs_new = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    for doc in index_docs_new:
        assert (doc in index) is False
