import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.qdrant.fixtures import start_storage, tmp_collection_name  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDoc):
    d: SimpleDoc


class DeepNestedDoc(BaseDoc):
    d: NestedDoc


class TorchDoc(BaseDoc):
    tens: TorchTensor[10]  # type: ignore[valid-type]


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_simple_schema(space):
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(space=space)  # type: ignore[valid-type]

    index = QdrantDocumentIndex[SimpleSchema](host='localhost')

    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(SimpleDoc(tens=np.ones(10)))
    index.index(index_docs)

    query = SimpleDoc(tens=np.ones(10))

    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5


def test_find_torch():
    index = QdrantDocumentIndex[TorchDoc](host='localhost')

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


@pytest.mark.tensorflow
def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10]  # type: ignore[valid-type]

    index = QdrantDocumentIndex[TfDoc](host='localhost')

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


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_flat_schema(space):
    class FlatSchema(BaseDoc):
        tens_one: NdArray = Field(dim=10, space=space)
        tens_two: NdArray = Field(dim=50, space=space)

    index = QdrantDocumentIndex[FlatSchema](host='localhost')

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

    # find on tens_two
    docs, scores = index.find(query, search_field='tens_two', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_nested_schema(space):
    class SimpleDoc(BaseDoc):
        tens: NdArray[10] = Field(space=space)  # type: ignore[valid-type]

    class NestedDoc(BaseDoc):
        d: SimpleDoc
        tens: NdArray[10] = Field(space=space)  # type: ignore[valid-type]

    class DeepNestedDoc(BaseDoc):
        d: NestedDoc
        tens: NdArray = Field(space=space, dim=10)

    index = QdrantDocumentIndex[DeepNestedDoc](host='localhost')

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

    # find on first nesting level
    docs, scores = index.find(query, search_field='d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5

    # find on second nesting level
    docs, scores = index.find(query, search_field='d__d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_batched(space, tmp_collection_name):  # noqa: F811
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(space=space)  # type: ignore[valid-type]

    index = QdrantDocumentIndex[SimpleSchema](
        host='localhost', collection_name=tmp_collection_name
    )

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

    docs, scores = index.find_batched(queries, search_field='tens', limit=1)

    assert len(docs) == 2
    assert len(docs[0]) == 1
    assert len(docs[1]) == 1
    assert len(scores) == 2
    assert len(scores[0]) == 1
    assert len(scores[1]) == 1
    assert docs[0][0].id == index_docs[0].id
    assert docs[1][0].id == index_docs[-1].id
