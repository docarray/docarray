import numpy as np
import pytest

from docarray import DocumentArray
from docarray.doc_index.backends.elastic_doc_index import ElasticDocumentIndex
from tests.doc_index.elastic.fixture import start_storage  # noqa: F401
from tests.doc_index.elastic.fixture import DeepNestedDoc, FlatDoc, NestedDoc, SimpleDoc


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


@pytest.fixture
def ten_deep_nested_docs():
    return [
        DeepNestedDoc(d=NestedDoc(d=SimpleDoc(tens=np.random.randn(10))))
        for _ in range(10)
    ]


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_simple_schema(ten_simple_docs, use_docarray):
    store = ElasticDocumentIndex[SimpleDoc]()
    if use_docarray:
        ten_simple_docs = DocumentArray[SimpleDoc](ten_simple_docs)

    store.index(ten_simple_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, use_docarray):
    store = ElasticDocumentIndex[FlatDoc]()
    if use_docarray:
        ten_flat_docs = DocumentArray[FlatDoc](ten_flat_docs)

    store.index(ten_flat_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(ten_nested_docs, use_docarray):
    store = ElasticDocumentIndex[NestedDoc]()
    if use_docarray:
        ten_nested_docs = DocumentArray[NestedDoc](ten_nested_docs)

    store.index(ten_nested_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_deep_nested_schema(ten_deep_nested_docs, use_docarray):
    store = ElasticDocumentIndex[DeepNestedDoc]()
    if use_docarray:
        ten_deep_nested_docs = DocumentArray[DeepNestedDoc](ten_deep_nested_docs)

    store.index(ten_deep_nested_docs)
    assert store.num_docs() == 10


def test_get_single(ten_simple_docs, ten_flat_docs, ten_nested_docs):
    # simple
    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    for d in ten_simple_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert np.all(store[id_].tens == d.tens)

    # flat
    store = ElasticDocumentIndex[FlatDoc]()
    store.index(ten_flat_docs)

    assert store.num_docs() == 10
    for d in ten_flat_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert np.all(store[id_].tens_one == d.tens_one)
        assert np.all(store[id_].tens_two == d.tens_two)

    # nested
    store = ElasticDocumentIndex[NestedDoc]()
    store.index(ten_nested_docs)

    assert store.num_docs() == 10
    for d in ten_nested_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert store[id_].d.id == d.d.id
        assert np.all(store[id_].d.tens == d.d.tens)


def test_get_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs):
    docs_to_get_idx = [0, 2, 4, 6, 8]

    # simple
    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens == d_in.tens)

    # flat
    store = ElasticDocumentIndex[FlatDoc]()
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
    store = ElasticDocumentIndex[NestedDoc]()
    store.index(ten_nested_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_nested_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert d_out.d.id == d_in.d.id
        assert np.all(d_out.d.tens == d_in.d.tens)


def test_get_key_error(ten_simple_docs):
    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    with pytest.raises(KeyError):
        store['not_a_real_id']


def test_del_single(ten_simple_docs):
    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)
    # delete once
    assert store.num_docs() == 10
    del store[ten_simple_docs[0].id]
    assert store.num_docs() == 9
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i == 0:  # deleted
            with pytest.raises(KeyError):
                store[id_]
        else:
            assert store[id_].id == id_
            assert np.all(store[id_].tens == d.tens)
    # delete again
    del store[ten_simple_docs[3].id]
    assert store.num_docs() == 8
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i in (0, 3):  # deleted
            with pytest.raises(KeyError):
                store[id_]
        else:
            assert store[id_].id == id_
            assert np.all(store[id_].tens == d.tens)


def test_del_multiple(ten_simple_docs):
    docs_to_del_idx = [0, 2, 4, 6, 8]

    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    docs_to_del = [ten_simple_docs[i] for i in docs_to_del_idx]
    ids_to_del = [d.id for d in docs_to_del]
    del store[ids_to_del]
    for i, doc in enumerate(ten_simple_docs):
        if i in docs_to_del_idx:
            with pytest.raises(KeyError):
                store[doc.id]
        else:
            assert store[doc.id].id == doc.id
            assert np.all(store[doc.id].tens == doc.tens)


def test_del_key_error(ten_simple_docs):
    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    with pytest.warns(UserWarning):
        del store['not_a_real_id']


def test_num_docs(ten_simple_docs):
    store = ElasticDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    assert store.num_docs() == 10

    del store[ten_simple_docs[0].id]
    assert store.num_docs() == 9

    del store[ten_simple_docs[3].id, ten_simple_docs[5].id]
    assert store.num_docs() == 7

    more_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(5)]
    store.index(more_docs)
    assert store.num_docs() == 12

    del store[more_docs[2].id, ten_simple_docs[7].id]
    assert store.num_docs() == 10