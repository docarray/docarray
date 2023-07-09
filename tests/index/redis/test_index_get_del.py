import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray
from tests.index.redis.fixtures import start_redis, tmp_index_name  # noqa: F401

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


@pytest.fixture
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


def test_num_docs(ten_simple_docs):
    index = RedisDocumentIndex[SimpleDoc](host='localhost')
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


def test_get_single(ten_simple_docs, tmp_index_name):
    index = RedisDocumentIndex[SimpleDoc](host='localhost', index_name=tmp_index_name)
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    doc_to_get = ten_simple_docs[3]
    doc_id = doc_to_get.id
    retrieved_doc = index[doc_id]
    assert retrieved_doc.id == doc_id
    assert np.allclose(retrieved_doc.tens, doc_to_get.tens)

    with pytest.raises(KeyError):
        index['some_id']


def test_get_multiple(ten_simple_docs, tmp_index_name):
    docs_to_get_idx = [0, 2, 4, 6, 8]
    index = RedisDocumentIndex[SimpleDoc](host='localhost', index_name=tmp_index_name)
    index.index(ten_simple_docs)

    assert index.num_docs() == 10
    docs_to_get = [ten_simple_docs[i] for i in docs_to_get_idx]
    ids_to_get = [d.id for d in docs_to_get]
    retrieved_docs = index[ids_to_get]
    for id_, d_in, d_out in zip(ids_to_get, docs_to_get, retrieved_docs):
        assert d_out.id == id_
        assert np.allclose(d_out.tens, d_in.tens)


def test_del_single(ten_simple_docs, tmp_index_name):
    index = RedisDocumentIndex[SimpleDoc](host='localhost', index_name=tmp_index_name)
    index.index(ten_simple_docs)
    assert index.num_docs() == 10

    doc_id = ten_simple_docs[3].id
    del index[doc_id]

    assert index.num_docs() == 9

    with pytest.raises(KeyError):
        index[doc_id]


def test_del_multiple(ten_simple_docs, tmp_index_name):
    docs_to_del_idx = [0, 2, 4, 6, 8]

    index = RedisDocumentIndex[SimpleDoc](host='localhost', index_name=tmp_index_name)
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
            assert np.allclose(index[doc.id].tens, doc.tens)


def test_contains(ten_simple_docs, tmp_index_name):
    index = RedisDocumentIndex[SimpleDoc](host='localhost', index_name=tmp_index_name)
    index.index(ten_simple_docs)

    for doc in ten_simple_docs:
        assert doc in index

    other_doc = SimpleDoc(tens=np.random.randn(10))
    assert other_doc not in index
