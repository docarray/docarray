from typing import Union

import numpy as np
import pytest

from docarray import BaseDoc, DocArray
from docarray.documents import ImageDoc, TextDoc
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray
from tests.integrations.doc_index.elastic.fixture import (  # noqa: F401
    DeepNestedDoc,
    FlatDoc,
    NestedDoc,
    SimpleDoc,
    start_storage_v8,
    ten_deep_nested_docs,
    ten_flat_docs,
    ten_nested_docs,
    ten_simple_docs,
)

pytestmark = [pytest.mark.slow, pytest.mark.index, pytest.mark.elasticv8]


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_simple_schema(ten_simple_docs, use_docarray):  # noqa: F811
    store = ElasticDocIndex[SimpleDoc]()
    if use_docarray:
        ten_simple_docs = DocArray[SimpleDoc](ten_simple_docs)

    store.index(ten_simple_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, use_docarray):  # noqa: F811
    store = ElasticDocIndex[FlatDoc]()
    if use_docarray:
        ten_flat_docs = DocArray[FlatDoc](ten_flat_docs)

    store.index(ten_flat_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(ten_nested_docs, use_docarray):  # noqa: F811
    store = ElasticDocIndex[NestedDoc]()
    if use_docarray:
        ten_nested_docs = DocArray[NestedDoc](ten_nested_docs)

    store.index(ten_nested_docs)
    assert store.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_deep_nested_schema(ten_deep_nested_docs, use_docarray):  # noqa: F811
    store = ElasticDocIndex[DeepNestedDoc]()
    if use_docarray:
        ten_deep_nested_docs = DocArray[DeepNestedDoc](ten_deep_nested_docs)

    store.index(ten_deep_nested_docs)
    assert store.num_docs() == 10


def test_get_single(ten_simple_docs, ten_flat_docs, ten_nested_docs):  # noqa: F811
    # simple
    store = ElasticDocIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    for d in ten_simple_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert np.all(store[id_].tens == d.tens)

    # flat
    store = ElasticDocIndex[FlatDoc]()
    store.index(ten_flat_docs)

    assert store.num_docs() == 10
    for d in ten_flat_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert np.all(store[id_].tens_one == d.tens_one)
        assert np.all(store[id_].tens_two == d.tens_two)

    # nested
    store = ElasticDocIndex[NestedDoc]()
    store.index(ten_nested_docs)

    assert store.num_docs() == 10
    for d in ten_nested_docs:
        id_ = d.id
        assert store[id_].id == id_
        assert store[id_].d.id == d.d.id
        assert np.all(store[id_].d.tens == d.d.tens)


def test_get_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs):  # noqa: F811
    docs_to_get_idx = [0, 2, 4, 6, 8]

    # simple
    store = ElasticDocIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens == d_in.tens)

    # flat
    store = ElasticDocIndex[FlatDoc]()
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
    store = ElasticDocIndex[NestedDoc]()
    store.index(ten_nested_docs)

    assert store.num_docs() == 10
    docs_to_get = [ten_nested_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = store[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert d_out.d.id == d_in.d.id
        assert np.all(d_out.d.tens == d_in.d.tens)


def test_get_key_error(ten_simple_docs):  # noqa: F811
    store = ElasticDocIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    with pytest.raises(KeyError):
        store['not_a_real_id']


def test_persisting(ten_simple_docs):  # noqa: F811
    store = ElasticDocIndex[SimpleDoc](index_name='test_persisting')
    store.index(ten_simple_docs)

    store2 = ElasticDocIndex[SimpleDoc](index_name='test_persisting')
    assert store2.num_docs() == 10


def test_del_single(ten_simple_docs):  # noqa: F811
    store = ElasticDocIndex[SimpleDoc]()
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


def test_del_multiple(ten_simple_docs):  # noqa: F811
    docs_to_del_idx = [0, 2, 4, 6, 8]

    store = ElasticDocIndex[SimpleDoc]()
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


def test_del_key_error(ten_simple_docs):  # noqa: F811
    store = ElasticDocIndex[SimpleDoc]()
    store.index(ten_simple_docs)

    with pytest.warns(UserWarning):
        del store['not_a_real_id']


def test_num_docs(ten_simple_docs):  # noqa: F811
    store = ElasticDocIndex[SimpleDoc]()
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


def test_index_union_doc():  # noqa: F811
    class MyDoc(BaseDoc):
        tensor: Union[NdArray, str]

    class MySchema(BaseDoc):
        tensor: NdArray

    store = ElasticDocIndex[MySchema]()
    doc = [MyDoc(tensor=np.random.randn(128))]
    store.index(doc)

    id_ = doc[0].id
    assert store[id_].id == id_
    assert np.all(store[id_].tensor == doc[0].tensor)


def test_index_multi_modal_doc():
    class MyMultiModalDoc(BaseDoc):
        image: ImageDoc
        text: TextDoc

    store = ElasticDocIndex[MyMultiModalDoc]()

    doc = [
        MyMultiModalDoc(
            image=ImageDoc(embedding=np.random.randn(128)), text=TextDoc(text='hello')
        )
    ]
    store.index(doc)

    id_ = doc[0].id
    assert store[id_].id == id_
    assert np.all(store[id_].image.embedding == doc[0].image.embedding)
    assert store[id_].text.text == doc[0].text.text


def test_elasticv7_version_check():
    with pytest.raises(ImportError):
        from docarray.index import ElasticV7DocIndex  # noqa: F401
