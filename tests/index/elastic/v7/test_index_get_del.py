from typing import Union

import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc, TextDoc
from docarray.index import ElasticV7DocIndex
from docarray.typing import NdArray
from tests.index.elastic.fixture import (  # noqa: F401
    DeepNestedDoc,
    FlatDoc,
    MyImageDoc,
    NestedDoc,
    SimpleDoc,
    start_storage_v7,
    ten_deep_nested_docs,
    ten_flat_docs,
    ten_nested_docs,
    ten_simple_docs,
)

pytestmark = [pytest.mark.slow, pytest.mark.index]


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_simple_schema(ten_simple_docs, use_docarray):  # noqa: F811
    index = ElasticV7DocIndex[SimpleDoc]()
    if use_docarray:
        ten_simple_docs = DocList[SimpleDoc](ten_simple_docs)

    index.index(ten_simple_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_flat_schema(ten_flat_docs, use_docarray):  # noqa: F811
    index = ElasticV7DocIndex[FlatDoc]()
    if use_docarray:
        ten_flat_docs = DocList[FlatDoc](ten_flat_docs)

    index.index(ten_flat_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_nested_schema(ten_nested_docs, use_docarray):  # noqa: F811
    index = ElasticV7DocIndex[NestedDoc]()
    if use_docarray:
        ten_nested_docs = DocList[NestedDoc](ten_nested_docs)

    index.index(ten_nested_docs)
    assert index.num_docs() == 10


@pytest.mark.parametrize('use_docarray', [True, False])
def test_index_deep_nested_schema(ten_deep_nested_docs, use_docarray):  # noqa: F811
    index = ElasticV7DocIndex[DeepNestedDoc]()
    if use_docarray:
        ten_deep_nested_docs = DocList[DeepNestedDoc](ten_deep_nested_docs)

    index.index(ten_deep_nested_docs)
    assert index.num_docs() == 10


def test_get_single(ten_simple_docs, ten_flat_docs, ten_nested_docs):  # noqa: F811
    # simple
    index = ElasticV7DocIndex[SimpleDoc]()
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    for d in ten_simple_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert np.all(index[id_].tens == d.tens)

    # flat
    index = ElasticV7DocIndex[FlatDoc]()
    index.index(ten_flat_docs)

    assert index.num_docs() == 10
    for d in ten_flat_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert np.all(index[id_].tens_one == d.tens_one)
        assert np.all(index[id_].tens_two == d.tens_two)

    # nested
    index = ElasticV7DocIndex[NestedDoc]()
    index.index(ten_nested_docs)

    assert index.num_docs() == 10
    for d in ten_nested_docs:
        id_ = d.id
        assert index[id_].id == id_
        assert index[id_].d.id == d.d.id
        assert np.all(index[id_].d.tens == d.d.tens)


def test_get_multiple(ten_simple_docs, ten_flat_docs, ten_nested_docs):  # noqa: F811
    docs_to_get_idx = [0, 2, 4, 6, 8]

    # simple
    index = ElasticV7DocIndex[SimpleDoc]()
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.all(d_out.tens == d_in.tens)

    # flat
    index = ElasticV7DocIndex[FlatDoc]()
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
    index = ElasticV7DocIndex[NestedDoc]()
    index.index(ten_nested_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_nested_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert d_out.d.id == d_in.d.id
        assert np.all(d_out.d.tens == d_in.d.tens)


def test_get_key_error(ten_simple_docs):  # noqa: F811
    index = ElasticV7DocIndex[SimpleDoc]()
    index.index(ten_simple_docs)

    with pytest.raises(KeyError):
        index['not_a_real_id']


def test_persisting(ten_simple_docs):  # noqa: F811
    index = ElasticV7DocIndex[SimpleDoc](index_name='test_persisting')
    index.index(ten_simple_docs)

    index2 = ElasticV7DocIndex[SimpleDoc](index_name='test_persisting')
    assert index2.num_docs() == 10


def test_del_single(ten_simple_docs):  # noqa: F811
    index = ElasticV7DocIndex[SimpleDoc]()
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


def test_del_multiple(ten_simple_docs):  # noqa: F811
    docs_to_del_idx = [0, 2, 4, 6, 8]

    index = ElasticV7DocIndex[SimpleDoc]()
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


def test_del_key_error(ten_simple_docs):  # noqa: F811
    index = ElasticV7DocIndex[SimpleDoc]()
    index.index(ten_simple_docs)

    with pytest.warns(UserWarning):
        del index['not_a_real_id']


def test_num_docs(ten_simple_docs):  # noqa: F811
    index = ElasticV7DocIndex[SimpleDoc]()
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


def test_index_union_doc():
    class MyDoc(BaseDoc):
        tensor: Union[NdArray, str]

    class MySchema(BaseDoc):
        tensor: NdArray

    index = ElasticV7DocIndex[MySchema]()
    doc = [MyDoc(tensor=np.random.randn(128))]
    index.index(doc)

    id_ = doc[0].id
    assert index[id_].id == id_
    assert np.all(index[id_].tensor == doc[0].tensor)


def test_index_multi_modal_doc():
    class MyMultiModalDoc(BaseDoc):
        image: MyImageDoc
        text: TextDoc

    index = ElasticV7DocIndex[MyMultiModalDoc]()

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
