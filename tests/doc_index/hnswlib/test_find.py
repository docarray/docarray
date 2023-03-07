import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.hnswlib_doc_index import HnswDocumentIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.doc_index]


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000)


class FlatDoc(BaseDocument):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDocument):
    d: SimpleDoc


class DeepNestedDoc(BaseDocument):
    d: NestedDoc


@pytest.mark.parametrize('space', ['cosine', 'l2', 'ip'])
def test_find_simple_schema(tmp_path, space):
    class SimpleSchema(BaseDocument):
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
def test_find_flat_schema(tmp_path, space):
    class FlatSchema(BaseDocument):
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
    class SimpleDoc(BaseDocument):
        tens: NdArray[10] = Field(space=space)

    class NestedDoc(BaseDocument):
        d: SimpleDoc
        tens: NdArray[10] = Field(space=space)

    class DeepNestedDoc(BaseDocument):
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
