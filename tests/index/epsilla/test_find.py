import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc, DocList
from docarray.index import EpsillaDocumentIndex
from docarray.typing import NdArray, TorchTensor
from tests.index.epsilla.common import epsilla_config
from tests.index.epsilla.fixtures import start_storage  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(is_embedding=True, dim=1000)  # type: ignore[valid-type]


class FlatDoc(BaseDoc):
    tens_one: NdArray = Field(is_embedding=True, dim=10)
    tens_two: NdArray = Field(dim=50)


class TorchDoc(BaseDoc):
    tens: TorchTensor[10] = Field(is_embedding=True)  # type: ignore[valid-type]


@pytest.mark.parametrize('space', ['l2', 'ip'])
def test_find_simple_schema(space, tmp_index_name):
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True, space=space)  # type: ignore[valid-type]

    index = EpsillaDocumentIndex[SimpleSchema](
        **epsilla_config, table_name=tmp_index_name
    )

    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(SimpleDoc(tens=np.ones(10)))
    index.index(index_docs)

    query = SimpleDoc(tens=np.ones(10))

    docs, scores = index.find(query, limit=5, search_field="tens")

    assert len(docs) == 5
    assert len(scores) == 5


def test_find_torch(tmp_index_name):
    index = EpsillaDocumentIndex[TorchDoc](**epsilla_config, table_name=tmp_index_name)

    index_docs = [TorchDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(TorchDoc(tens=np.ones(10)))
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = TorchDoc(tens=np.ones(10))

    result_docs, scores = index.find(query, limit=5, search_field="tens")

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TorchTensor)


@pytest.mark.tensorflow
def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10] = Field(is_embedding=True)  # type: ignore[valid-type]

    index = EpsillaDocumentIndex[TfDoc](**epsilla_config)

    index_docs = [TfDoc(tens=np.random.rand(10)) for _ in range(10)]
    index.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = index_docs[-1]
    docs, scores = index.find(query, limit=5, search_field="tens")

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)


def test_find_batched(tmp_index_name):  # noqa: F811
    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = EpsillaDocumentIndex[SimpleSchema](
        **epsilla_config, table_name=tmp_index_name
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

    docs, scores = index.find_batched(queries, limit=1, search_field="tens")

    assert len(docs) == 2
    assert len(docs[0]) == 1
    assert len(docs[1]) == 1
    assert len(scores) == 2
    assert len(scores[0]) == 1
    assert len(scores[1]) == 1


def test_contain(tmp_index_name):
    class SimpleDoc(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    class SimpleSchema(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = EpsillaDocumentIndex[SimpleSchema](
        **epsilla_config, table_name=tmp_index_name
    )
    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]

    assert (index_docs[0] in index) is False

    index.index(index_docs)

    for doc in index_docs:
        assert (doc in index) is True

    index_docs_new = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    for doc in index_docs_new:
        assert (doc in index) is False


@pytest.mark.parametrize('space', ['l2', 'ip'])
def test_find_flat_schema(space, tmp_index_name):
    class FlatSchema(BaseDoc):
        tens_one: NdArray[10] = Field(space=space, is_embedding=True)
        tens_two: NdArray[50] = Field(space=space)

    index = EpsillaDocumentIndex[FlatSchema](
        **epsilla_config, table_name=tmp_index_name
    )

    index_docs = [
        FlatDoc(tens_one=np.zeros(10), tens_two=np.zeros(50)) for _ in range(10)
    ]
    index_docs.append(FlatDoc(tens_one=np.zeros(10), tens_two=np.ones(50)))
    index_docs.append(FlatDoc(tens_one=np.ones(10), tens_two=np.zeros(50)))
    index.index(index_docs)

    query = FlatDoc(tens_one=np.ones(10), tens_two=np.ones(50))

    # find on tens_one
    docs, scores = index.find(query, limit=5, search_field="tens_one")
    assert len(docs) == 5
    assert len(scores) == 5


def test_find_nested_schema(tmp_index_name):
    class SimpleDoc(BaseDoc):
        tens: NdArray[10]  # type: ignore[valid-type]

    class NestedDoc(BaseDoc):
        d: SimpleDoc
        tens: NdArray[10]  # type: ignore[valid-type]

    class DeepNestedDoc(BaseDoc):
        d: NestedDoc
        tens: NdArray[10] = Field(is_embedding=True)

    index = EpsillaDocumentIndex[DeepNestedDoc](
        **epsilla_config, table_name=tmp_index_name
    )

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
    docs, scores = index.find(query, limit=5, search_field="tens")
    assert len(docs) == 5
    assert len(scores) == 5


def test_find_empty_index(tmp_index_name):
    empty_index = EpsillaDocumentIndex[SimpleDoc](
        **epsilla_config, table_name=tmp_index_name
    )
    query = SimpleDoc(tens=np.random.rand(10))

    # find
    docs, scores = empty_index.find(query, limit=5, search_field="tens")
    assert len(docs) == 0
    assert len(scores) == 0

    # find_batched
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
    docs, scores = empty_index.find_batched(queries, limit=1, search_field="tens")

    assert len(docs) == 0
    assert len(scores) == 0


def test_simple_usage(tmp_index_name):
    class MyDoc(BaseDoc):
        text: str
        embedding: NdArray[128] = Field(is_embedding=True)

    docs = [MyDoc(text='hey', embedding=np.random.rand(128)) for _ in range(200)]
    queries = docs[0:3]
    index = EpsillaDocumentIndex[MyDoc](**epsilla_config, table_name=tmp_index_name)
    index.index(docs=DocList[MyDoc](docs))
    resp = index.find_batched(queries=queries, limit=5, search_field="embedding")
    docs_responses = resp.documents
    assert len(docs_responses) == 3
    for q, matches in zip(queries, docs_responses):
        assert len(matches) == 5
        assert q.id == matches[0].id


def test_filter_range(tmp_index_name):  # noqa: F811
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='l2', is_embedding=True)  # type: ignore[valid-type]
        number: int

    index = EpsillaDocumentIndex[SimpleSchema](
        **epsilla_config, table_name=tmp_index_name
    )

    docs = index.filter("number > 8", limit=5)
    assert len(docs) == 0

    index_docs = [
        SimpleSchema(
            embedding=np.zeros(10),
            number=i,
        )
        for i in range(10)
    ]
    index.index(index_docs)

    docs = index.filter("number > 8", limit=5)

    assert len(docs) == 1

    docs = index.filter(f"id = '{index_docs[0].id}'", limit=5)
    assert docs[0].id == index_docs[0].id


def test_query_builder(tmp_index_name):
    class SimpleSchema(BaseDoc):
        tensor: NdArray[10] = Field(is_embedding=True)
        price: int

    db = EpsillaDocumentIndex[SimpleSchema](**epsilla_config, table_name=tmp_index_name)

    index_docs = [
        SimpleSchema(tensor=np.array([i + 1] * 10), price=i + 1) for i in range(10)
    ]
    db.index(index_docs)

    q = (
        db.build_query()
        .find(query=np.ones(10), search_field="tensor")
        .filter(filter_query='price <= 3')
        .build(limit=5)
    )

    docs = db.execute_query(q)

    assert len(docs) == 3
    for doc in docs:
        assert doc.price <= 3
