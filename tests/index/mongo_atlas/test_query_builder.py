import numpy as np
import pytest

from .fixtures import *  # noqa: F403
from .helpers import assert_when_ready


def test_find_uses_provided_vector(simple_index):  # noqa: F811
    index = simple_index

    query = (
        index.build_query().find(query=np.ones(10), search_field='embedding').build(7)
    )

    assert query.vector_field == 'embedding'
    assert np.allclose(query.vector_query, np.ones(10))
    assert query.filters == []
    assert query.limit == 7


def test_multiple_find_returns_averaged_vector(simple_index):  # noqa: F811
    index = simple_index

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=np.ones(10), search_field='embedding')
        .find(query=np.zeros(10), search_field='embedding')
        .build(5)
    )

    assert query.vector_field == 'embedding'
    assert np.allclose(query.vector_query, np.array([0.5] * 10))
    assert query.filters == []
    assert query.limit == 5


def test_filter_passes_filter(simple_index):  # noqa: F811
    index = simple_index

    filter = {"number": {"$lt": 1}}
    query = index.build_query().filter(query=filter).build(11)  # type: ignore[attr-defined]

    assert query.vector_field is None
    assert query.vector_query is None
    assert query.filters == [{"query": filter}]
    assert query.limit == 11


def test_text_search_filter(simple_index):  # noqa: F811
    index = simple_index

    kwargs = dict(query='lorem ipsum', search_field='text')
    with pytest.raises(NotImplementedError):
        index.build_query().text_search(**kwargs).build(3)  # type: ignore[attr-defined]


def test_query_builder_execute_query_find_filter(
    simple_index_with_docs,  # noqa: F811
):
    index, docs = simple_index_with_docs

    find_query = np.ones(10)
    filter_query1 = {"number": {"$lt": 8}}
    filter_query2 = {"number": {"$gt": 5}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .filter(query=filter_query1)
        .filter(query=filter_query2)
        .build(limit=5)
    )

    def pred():
        docs = index.execute_query(query)

        assert len(docs.documents) == 2
        assert set(docs.documents.number) == {6, 7}

    assert_when_ready(pred)


def test_query_builder_execute_only_filter(
    simple_index_with_docs,  # noqa: F811
):
    index, docs = simple_index_with_docs

    filter_query1 = {"number": {"$lt": 8}}
    filter_query2 = {"number": {"$gt": 5}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .filter(query=filter_query1)
        .filter(query=filter_query2)
        .build(limit=5)
    )

    def pred():
        docs = index.execute_query(query)

        assert len(docs.documents) == 2
        assert set(docs.documents.number) == {6, 7}

    assert_when_ready(pred)


def test_query_builder_execute_only_filter_text(
    simple_index_with_docs,  # noqa: F811
):
    index, docs = simple_index_with_docs

    filter_query1 = {"number": {"$eq": 0}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .text_search(query="Python is a valuable skill", search_field='text')
        .filter(query=filter_query1)
        .build(limit=5)
    )

    def pred():
        docs = index.execute_query(query)

        assert len(docs.documents) == 1
        assert set(docs.documents.number) == {0}

    assert_when_ready(pred)


def test_query_builder_hybrid_search(
    simple_index_with_docs,  # noqa: F811
):
    find_query = np.ones(10)
    # filter_query1 = {"number": {"$gt": 0}}
    index, docs = simple_index_with_docs

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        # .filter(query=filter_query1)
        .build(limit=10)
    )

    def pred():
        docs = index.execute_query(query)

        assert len(docs.documents) == 10
        assert set(docs.documents.number) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    assert_when_ready(pred)
