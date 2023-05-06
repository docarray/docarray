from typing import Union

import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc, TextDoc
from docarray.index import ElasticDocIndex
from docarray.typing import NdArray
from tests.index.elastic.fixture import (  # noqa: F401
    DeepNestedDoc,
    FlatDoc,
    MyImageDoc,
    NestedDoc,
    SimpleDoc,
    start_storage_v8,
    ten_deep_nested_docs,
    ten_flat_docs,
    ten_nested_docs,
    ten_simple_docs,
    tmp_index_name,
)

pytestmark = [pytest.mark.slow, pytest.mark.index, pytest.mark.elasticv8]


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_simple_schema(
    ten_simple_docs, use_docarray, tmp_index_name  # noqa: F811
):
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    if use_docarray:
        ten_simple_docs = DocList[SimpleDoc](ten_simple_docs)

    index.index(ten_simple_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, use_docarray, tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[FlatDoc](index_name=tmp_index_name)
    if use_docarray:
        ten_flat_docs = DocList[FlatDoc](ten_flat_docs)

    index.index(ten_flat_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(
    ten_nested_docs, use_docarray, tmp_index_name  # noqa: F811
):
    index = ElasticDocIndex[NestedDoc](index_name=tmp_index_name)
    if use_docarray:
        ten_nested_docs = DocList[NestedDoc](ten_nested_docs)

    index.index(ten_nested_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_deep_nested_schema(
    ten_deep_nested_docs, use_docarray, tmp_index_name  # noqa: F811
):
    index = ElasticDocIndex[DeepNestedDoc](index_name=tmp_index_name)
    if use_docarray:
        ten_deep_nested_docs = DocList[DeepNestedDoc](ten_deep_nested_docs)

    index.index(ten_deep_nested_docs)
    assert index.num_docs() == 10


def test_get_single(ten_simple_docs, ten_flat_docs, ten_nested_docs):  # noqa: F811
    # simple
    index = ElasticDocIndex[SimpleDoc]()
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    for d in ten_simple_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert np.all(index[id_].tens == d.tens)
    index._client.indices.delete(index='simpledoc', ignore_unavailable=True)

    # flat
    index = ElasticDocIndex[FlatDoc]()
    index.index(ten_flat_docs)

    assert index.num_docs() == 10
    for d in ten_flat_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert np.all(index[id_].tens_one == d.tens_one)
        assert np.all(index[id_].tens_two == d.tens_two)
    index._client.indices.delete(index='flatdoc', ignore_unavailable=True)

    # nested
    index = ElasticDocIndex[NestedDoc]()
    index.index(ten_nested_docs)

    assert index.num_docs() == 10
    for d in ten_nested_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert index[id_].d.id == d.d.id
        assert np.all(index[id_].d.tens == d.d.tens)
    index._client.indices.delete(index='nesteddoc', ignore_unavailable=True)


def test_get_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs):  # noqa: F811
    docs_to_get_idx = [0, 2, 4, 6, 8]

    # simple
    index = ElasticDocIndex[SimpleDoc]()
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens == d_in.tens)

    # flat
    index = ElasticDocIndex[FlatDoc]()
    index.index(ten_flat_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_flat_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens_one == d_in.tens_one)
        assert np.all(d_out.tens_two == d_in.tens_two)

    # nested
    index = ElasticDocIndex[NestedDoc]()
    index.index(ten_nested_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_nested_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert d_out.d.id == d_in.d.id
        assert np.all(d_out.d.tens == d_in.d.tens)


def test_get_key_error(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    with pytest.raises(KeyError):
        index['not_a_real_id']


def test_persisting(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    index2 = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    assert index2.num_docs() == 10


def test_del_single(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)
    # delete once
    assert index.num_docs() == 10
    del index[ten_simple_docs[0].id]
    assert index.num_docs() == 9
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i == 0:  # deleted
            with pytest.raises(KeyError):
                index[id_]
        else:
            assert index[id_].id == id_
            assert np.all(index[id_].tens == d.tens)
    # delete again
    del index[ten_simple_docs[3].id]
    assert index.num_docs() == 8
    for i, d in enumerate(ten_simple_docs):
        id_ = d.id
        if i in (0, 3):  # deleted
            with pytest.raises(KeyError):
                index[id_]
        else:
            assert index[id_].id == id_
            assert np.all(index[id_].tens == d.tens)


def test_del_multiple(ten_simple_docs, tmp_index_name):  # noqa: F811
    docs_to_del_idx = [0, 2, 4, 6, 8]

    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    docs_to_del = [ten_simple_docs[i] for i in docs_to_del_idx]
    ids_to_del = [d.id for d in docs_to_del]
    del index[ids_to_del]
    for i, doc in enumerate(ten_simple_docs):
        if i in docs_to_del_idx:
            with pytest.raises(KeyError):
                index[doc.id]
        else:
            assert index[doc.id].id == doc.id
            assert np.all(index[doc.id].tens == doc.tens)


def test_del_key_error(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    with pytest.warns(UserWarning):
        del index['not_a_real_id']


def test_num_docs(ten_simple_docs, tmp_index_name):  # noqa: F811
    index = ElasticDocIndex[SimpleDoc](index_name=tmp_index_name)
    index.index(ten_simple_docs)

    assert index.num_docs() == 10

    del index[ten_simple_docs[0].id]
    assert index.num_docs() == 9

    del index[ten_simple_docs[3].id, ten_simple_docs[5].id]
    assert index.num_docs() == 7

    more_docs = [SimpleDoc(tens=np.random.rand(10)) for _ in range(5)]
    index.index(more_docs)
    assert index.num_docs() == 12

    del index[more_docs[2].id, ten_simple_docs[7].id]
    assert index.num_docs() == 10


def test_index_union_doc():  # noqa: F811
    class MyDoc(BaseDoc):
        tensor: Union[NdArray, str]

    class MySchema(BaseDoc):
        tensor: NdArray[128]

    index = ElasticDocIndex[MySchema]()
    doc = [MyDoc(tensor=np.random.randn(128))]
    index.index(doc)

    id_ = doc[0].id
    assert index[id_].id == id_
    assert np.all(index[id_].tensor == doc[0].tensor)


def test_index_multi_modal_doc():
    class MyMultiModalDoc(BaseDoc):
        image: MyImageDoc
        text: TextDoc

    index = ElasticDocIndex[MyMultiModalDoc]()

    doc = [
        MyMultiModalDoc(
            image=ImageDoc(embedding=np.random.randn(128)), text=TextDoc(text='hello')
        )
    ]
    index.index(doc)

    id_ = doc[0].id
    assert index[id_].id == id_
    assert np.all(index[id_].image.embedding == doc[0].image.embedding)
    assert index[id_].text.text == doc[0].text.text

    query = doc[0]
    docs, _ = index.find(query, limit=10, search_field='image__embedding')
    assert len(docs) > 0


def test_elasticv7_version_check():
    with pytest.raises(ImportError):
        from docarray.index import ElasticV7DocIndex

        _ = ElasticV7DocIndex[SimpleDoc]()
