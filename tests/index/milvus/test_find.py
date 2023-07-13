import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.milvus.fixtures import start_storage  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(is_embedding=True, dim=1000)  # type: ignore[valid-type]


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(is_embedding=True, dim=10)
    tens_two: NdArray = Field(dim=50)


class TorchDoc(BaseDoc):
    tens: TorchTensor[10] = Field(is_embedding=True)  # type: ignore[valid-type]


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_simple_schema(space):
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True, space=space)  # type: ignore[valid-type]

    index = MilvusDocumentIndex[SimpleSchema]()

    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(SimpleDoc(tens=np.ones(10)))
    index.index(index_docs)

    query = SimpleDoc(tens=np.ones(10))

    docs, scores = index.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5


def test_find_torch():
    index = MilvusDocumentIndex[TorchDoc]()

    index_docs = [TorchDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(TorchDoc(tens=np.ones(10)))
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = TorchDoc(tens=np.ones(10))

    result_docs, scores = index.find(query, limit=5)

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TorchTensor)


@pytest.mark.tensorflow
def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10] = Field(is_embedding=True)  # type: ignore[valid-type]

    index = MilvusDocumentIndex[TfDoc]()

    index_docs = [TfDoc(tens=np.random.rand(10)) for _ in range(10)]
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = index_docs[-1]
    docs, scores = index.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)


def test_find_batched():  # noqa: F811
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = MilvusDocumentIndex[SimpleSchema]()

    index_docs = [SimpleDoc(tens=vector) for vector in np.identity(10)]
    index.index(index_docs)

    queries = DocList[SimpleDoc](
        [
            SimpleDoc(
                tens=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ),
            SimpleDoc(
                tens=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1])
            ),
        ]
    )

    docs, scores = index.find_batched(queries, limit=1)

    assert len(docs) == 2
    assert len(docs[0]) == 1
    assert len(docs[1]) == 1
    assert len(scores) == 2
    assert len(scores[0]) == 1
    assert len(scores[1]) == 1


def test_contain():
    class SimpleDoc(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = MilvusDocumentIndex[SimpleSchema]()
    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]

    assert (index_docs[0] in index) is False

    index.index(index_docs)

    for doc in index_docs:
        assert (doc in index) is True

    index_docs_new = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    for doc in index_docs_new:
        assert (doc in index) is False


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_flat_schema(space):
    class FlatSchema(BaseDoc):
        tens_one: NdArray[10] = Field(space=space, is_embedding=True)
        tens_two: NdArray[50] = Field(space=space)

    index = MilvusDocumentIndex[FlatSchema]()

    index_docs = [
        FlatDoc(tens_one=np.zeros(10), tens_two=np.zeros(50)) for _ in range(10)
    ]
    index_docs.append(FlatDoc(tens_one=np.zeros(10), tens_two=np.ones(50)))
    index_docs.append(FlatDoc(tens_one=np.ones(10), tens_two=np.zeros(50)))
    index.index(index_docs)

    query = FlatDoc(tens_one=np.ones(10), tens_two=np.ones(50))

    # find on tens_one
    docs, scores = index.find(query, limit=5)
    assert len(docs) == 5
    assert len(scores) == 5


def test_find_nested_schema():
    class SimpleDoc(BaseDoc):
        tens: NdArray[10]  # type: ignore[valid-type]

    class NestedDoc(BaseDoc):
        d: SimpleDoc
        tens: NdArray[10]  # type: ignore[valid-type]

    class DeepNestedDoc(BaseDoc):
        d: NestedDoc
        tens: NdArray[10] = Field(is_embedding=True)

    index = MilvusDocumentIndex[DeepNestedDoc]()

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

    # find on root level (only support one level now)
    docs, scores = index.find(query, limit=5)
    assert len(docs) == 5
    assert len(scores) == 5


def test_find_empty_index():
    empty_index = MilvusDocumentIndex[SimpleDoc]()
    query = SimpleDoc(tens=np.random.rand(10))

    docs, scores = empty_index.find(query, limit=5)
    assert len(docs) == 0
    assert len(scores) == 0


def test_simple_usage():
    class MyDoc(BaseDoc):
        text: str
        embedding: NdArray[128] = Field(is_embedding=True)

    docs = [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    queries = docs[0:3]
    index = MilvusDocumentIndex[MyDoc]()
    index.index(docs=DocList[MyDoc](docs))
    print('num docs', index.num_docs())
    resp = index.find_batched(queries=queries, limit=5)
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 5
        assert q.id == matches[0].id
