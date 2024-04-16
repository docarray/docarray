import numpy as np

from .fixtures import *  # noqa: F403
from .helpers import assert_when_ready


def test_find_uses_provided_vector(simple_index):  # noqa: F811
    index = simple_index

    query = (
        index.build_query().find(query=np.ones(10), search_field='embedding').build(7)
    )

    query_vector = query.vector_fields.pop('embedding')
    assert query.vector_fields == {}
    assert np.allclose(query_vector, np.ones(10))
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

    query_vector = query.vector_fields.pop('embedding')
    assert query.vector_fields == {}
    assert np.allclose(query_vector, np.array([0.5] * 10))
    assert query.filters == []
    assert query.limit == 5


def test_filter_passes_filter(simple_index):  # noqa: F811
    index = simple_index

    filter = {"number": {"$lt": 1}}
    query = index.build_query().filter(query=filter).build(11)  # type: ignore[attr-defined]

    assert query.vector_fields == {}
    assert query.filters == [{"query": filter}]
    assert query.limit == 11


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
    index, docs = simple_index_with_docs

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        .build(limit=10)
    )

    def pred():
        docs = index.execute_query(query)

        assert len(docs.documents) == 10
        assert set(docs.documents.number) == {4, 5, 7, 8, 0, 6, 2, 9, 1, 3}

    assert_when_ready(pred)
