import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray, TorchTensor
from docarray.utils._internal.misc import is_tf_available

tf_available = is_tf_available()

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

    store = HnswDocumentIndex[SimpleSchema](work_dir=str(tmp_path))

    index_docs = [SimpleDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(SimpleDoc(tens=np.ones(10)))
    store.index(index_docs)

    query = SimpleDoc(tens=np.ones(10))

    docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-1].id
    assert np.allclose(docs[0].tens, index_docs[-1].tens)
    for result in docs[1:]:
        assert np.allclose(result.tens, np.zeros(10))


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_torch(tmp_path, space):
    store = HnswDocumentIndex[TorchDoc](work_dir=str(tmp_path))

    index_docs = [TorchDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(TorchDoc(tens=np.ones(10)))
    store.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TorchTensor)

    query = TorchDoc(tens=np.ones(10))

    result_docs, scores = store.find(query, search_field='tens', limit=5)

    assert len(result_docs) == 5
    assert len(scores) == 5
    for doc in result_docs:
        assert isinstance(doc.tens, TorchTensor)
    assert result_docs[0].id == index_docs[-1].id
    assert torch.allclose(result_docs[0].tens, index_docs[-1].tens)
    for result in result_docs[1:]:
        assert torch.allclose(result.tens, torch.zeros(10, dtype=torch.float64))


@pytest.mark.skipif(not tf_available, reason="Tensorflow not found")
@pytest.mark.tensorflow
def test_find_tensorflow(tmp_path):
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10]

    store = HnswDocumentIndex[TfDoc](work_dir=str(tmp_path))

    index_docs = [TfDoc(tens=np.zeros(10)) for _ in range(10)]
    index_docs.append(TfDoc(tens=np.ones(10)))
    store.index(index_docs)

    for doc in index_docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    query = TfDoc(tens=np.ones(10))

    result_docs, scores = store.find(query, search_field='tens', limit=5)

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

    store = HnswDocumentIndex[FlatSchema](work_dir=str(tmp_path))

    index_docs = [
        FlatDoc(tens_one=np.zeros(10), tens_two=np.zeros(50)) for _ in range(10)
    ]
    index_docs.append(FlatDoc(tens_one=np.zeros(10), tens_two=np.ones(50)))
    index_docs.append(FlatDoc(tens_one=np.ones(10), tens_two=np.zeros(50)))
    store.index(index_docs)

    query = FlatDoc(tens_one=np.ones(10), tens_two=np.ones(50))

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

    store = HnswDocumentIndex[DeepNestedDoc](work_dir=str(tmp_path))

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
    store.index(index_docs)

    query = DeepNestedDoc(
        d=NestedDoc(d=SimpleDoc(tens=np.ones(10)), tens=np.ones(10)), tens=np.ones(10)
    )

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
    assert docs[0].id == index_docs[-2].id
    assert np.allclose(docs[0].d.tens, index_docs[-2].d.tens)

    # find on second nesting level
    docs, scores = store.find(query, search_field='d__d__tens', limit=5)
    assert len(docs) == 5
    assert len(scores) == 5
    assert docs[0].id == index_docs[-3].id
    assert np.allclose(docs[0].d.d.tens, index_docs[-3].d.d.tens)
