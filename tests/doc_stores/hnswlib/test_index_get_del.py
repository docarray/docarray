import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument, DocumentArray
from docarray.storage.backends.HnswDocStore import HnswDocumentStore
from docarray.typing import NdArray


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000)


class FlatDoc(BaseDocument):
    tens_one: NdArray = Field(dim=10)
    tens_two: NdArray = Field(dim=50)


class NestedDoc(BaseDocument):
    d: SimpleDoc


class DeepNestedDoc(BaseDocument):
    d: NestedDoc


@pytest.fixture
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


@pytest.fixture
def ten_flat_docs():
    return [
        FlatDoc(tens_one=np.random.randn(10), tens_two=np.random.randn(50))
        for _ in range(10)
    ]


@pytest.fixture
def ten_nested_docs():
    return [NestedDoc(d=SimpleDoc(tens=np.random.randn(10))) for _ in range(10)]


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_simple_schema(ten_simple_docs, tmp_path, use_docarray):
    store = HnswDocumentStore[SimpleDoc](work_dir=str(tmp_path))
    if use_docarray:
        ten_simple_docs = DocumentArray[SimpleDoc](ten_simple_docs)

    store.index(ten_simple_docs)
    assert store.num_docs() == 10
    for index in store._hnsw_indices.values():
        assert index.get_current_count() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, tmp_path, use_docarray):
    store = HnswDocumentStore[FlatDoc](work_dir=str(tmp_path))
    if use_docarray:
        ten_flat_docs = DocumentArray[FlatDoc](ten_flat_docs)

    store.index(ten_flat_docs)
    assert store.num_docs() == 10
    for index in store._hnsw_indices.values():
        assert index.get_current_count() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(ten_nested_docs, tmp_path, use_docarray):
    store = HnswDocumentStore[NestedDoc](work_dir=str(tmp_path))
    if use_docarray:
        ten_nested_docs = DocumentArray[NestedDoc](ten_nested_docs)

    store.index(ten_nested_docs)
    assert store.num_docs() == 10
    for index in store._hnsw_indices.values():
        assert index.get_current_count() == 10


def test_get_single(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    simple_path = tmp_path / 'simple'
    flat_path = tmp_path / 'flat'
    nested_path = tmp_path / 'nested'

    # simple
    store = HnswDocumentStore[SimpleDoc](work_dir=str(simple_path))
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    for d in ten_simple_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert np.all(store[id_].tens == d.tens)

    # flat
    store = HnswDocumentStore[FlatDoc](work_dir=str(flat_path))
    store.index(ten_flat_docs)

    assert store.num_docs() == 10
    for d in ten_flat_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert np.all(store[id_].tens_one == d.tens_one)
        assert np.all(store[id_].tens_two == d.tens_two)

    # nested
    store = HnswDocumentStore[NestedDoc](work_dir=str(nested_path))
    store.index(ten_nested_docs)

    assert store.num_docs() == 10
    for d in ten_nested_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert store[id_].d.id == d.d.id
        assert np.all(store[id_].d.tens == d.d.tens)


def test_get_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs, tmp_path):
    simple_path = tmp_path / 'simple'
    flat_path = tmp_path / 'flat'
    nested_path = tmp_path / 'nested'
    docs_to_get_idx = [0, 2, 4, 6, 8]

    # simple
    store = HnswDocumentStore[SimpleDoc](work_dir=str(simple_path))
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens == d_in.tens)

    # flat
    store = HnswDocumentStore[FlatDoc](work_dir=str(flat_path))
    store.index(ten_flat_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_flat_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens_one == d_in.tens_one)
        assert np.all(d_out.tens_two == d_in.tens_two)

    # nested
    store = HnswDocumentStore[NestedDoc](work_dir=str(nested_path))
    store.index(ten_nested_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_nested_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert d_out.d.id == d_in.d.id
        assert np.all(d_out.d.tens == d_in.d.tens)
