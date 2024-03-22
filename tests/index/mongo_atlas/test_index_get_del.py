import numpy as np
import pytest

from .fixtures import *  # noqa
from .helpers import assert_when_ready

N_DIM = 10


def test_num_docs(simple_index_with_docs, simple_schema):
    index, docs = simple_index_with_docs
    query = np.ones(N_DIM)

    def check_n_elements(n):
        def pred():
            return index.num_docs() == 10

        return pred

    assert_when_ready(check_n_elements(10))

    del index[docs[0].id]

    assert_when_ready(check_n_elements(9))

    del index[docs[3].id, docs[5].id]

    assert_when_ready(check_n_elements(7))

    elems = [simple_schema(embedding=query, text="other", number=10) for _ in range(3)]
    index.index(elems)

    assert_when_ready(check_n_elements(10))

    del index[elems[0].id, elems[1].id]

    def check_ramaining_ids():
        assert index.num_docs() == 8
        # get everything
        elem_ids = set(
            doc.id
            for doc in index.find(query, search_field='embedding', limit=30).documents
        )
        expected_ids = {doc.id for i, doc in enumerate(docs) if i not in (3, 5, 0)}
        expected_ids.add(elems[2].id)
        assert elem_ids == expected_ids

    assert_when_ready(check_ramaining_ids)


def test_get_single(simple_index_with_docs):

    index, docs = simple_index_with_docs

    expected_doc = docs[5]
    retrieved_doc = index[expected_doc.id]

    assert retrieved_doc.id == expected_doc.id
    assert np.allclose(retrieved_doc.embedding, expected_doc.embedding)

    with pytest.raises(KeyError):
        index['An id that does not exist']


def test_get_multiple(simple_index_with_docs):
    index, docs = simple_index_with_docs

    # get the odd documents
    docs_to_get = [doc for i, doc in enumerate(docs) if i % 2 == 1]
    retrieved_docs = index[[doc.id for doc in docs_to_get]]
    assert set(doc.id for doc in docs_to_get) == set(doc.id for doc in retrieved_docs)


def test_del_single(simple_index_with_docs):
    index, docs = simple_index_with_docs
    del index[docs[1].id]

    def pred():
        assert index.num_docs() == 9

    assert_when_ready(pred)

    with pytest.raises(KeyError):
        index[docs[1].id]


def test_del_multiple(simple_index_with_docs):
    index, docs = simple_index_with_docs

    # get the odd documents
    docs_to_del = [doc for i, doc in enumerate(docs) if i % 2 == 1]

    del index[[d.id for d in docs_to_del]]
    for i, doc in enumerate(docs):
        if i % 2 == 1:
            with pytest.raises(KeyError):
                index[doc.id]
        else:
            assert index[doc.id].id == doc.id
            assert np.allclose(index[doc.id].embedding, doc.embedding)


def test_contains(simple_index_with_docs, simple_schema):
    index, docs = simple_index_with_docs

    for doc in docs:
        assert doc in index

    other_doc = simple_schema(embedding=[1.0] * N_DIM, text="other", number=10)
    assert other_doc not in index
