import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray, TorchTensor

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class TorchDoc(BaseDoc):
    tens: TorchTensor[10]  # type: ignore[valid-type]


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_simple_schema(space):
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(space=space)  # type: ignore[valid-type]

    index = MilvusDocumentIndex[SimpleSchema](index_name="tens")

    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(SimpleDoc(tens=np.ones(10)))
    index.index(index_docs)

    query = SimpleDoc(tens=np.ones(10))

    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5


def test_find_torch():
    index = MilvusDocumentIndex[TorchDoc](index_name="tens")

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

    index = MilvusDocumentIndex[TfDoc](index_name="tens")

    index_docs = [TfDoc(tens=np.random.rand(10)) for _ in range(10)]
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = index_docs[-1]
    docs, scores = index.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)


def test_find_batched():  # noqa: F811
    class SimpleSchema(BaseDoc):
        tens: NdArray[10]

    index = MilvusDocumentIndex[SimpleSchema](index_name="tens")

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


def test_contain():
    class SimpleDoc(BaseDoc):
        tens: NdArray[10]

    class SimpleSchema(BaseDoc):
        tens: NdArray[10]

    index = MilvusDocumentIndex[SimpleSchema](index_name="tens")
    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]

    assert (index_docs[0] in index) is False

    index.index(index_docs)

    for doc in index_docs:
        assert (doc in index) is True

    index_docs_new = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    for doc in index_docs_new:
        assert (doc in index) is False
