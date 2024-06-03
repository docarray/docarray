import numpy as np
import pytest

from . import assert_when_ready


def test_missing_required_var_exceptions(simple_index):  # noqa: F811
    """Ensure that exceptions are raised when required arguments are not provided."""

    with pytest.raises(ValueError):
        simple_index.build_query().find().build()

    with pytest.raises(ValueError):
        simple_index.build_query().text_search().build()

    with pytest.raises(ValueError):
        simple_index.build_query().filter().build()


def test_find_uses_provided_vector(simple_index):  # noqa: F811
    query = (
        simple_index.build_query().find(query=np.ones(10), search_field='embedding').build(7)
    )

    query_vector = query.vector_fields.pop('embedding')
    assert query.vector_fields == {}
    assert np.allclose(query_vector, np.ones(10))
    assert query.filters == []
    assert query.limit == 7


def test_multiple_find_returns_averaged_vector(simple_index, n_dim):  # noqa: F811
    query = (
        simple_index.build_query()  # type: ignore[attr-defined]
        .find(query=np.ones(n_dim), search_field='embedding')
        .find(query=np.zeros(n_dim), search_field='embedding')
        .build(5)
    )

    assert len(query.vector_fields) == 1
    query_vector = query.vector_fields.pop('embedding')
    assert query.vector_fields == {}
    assert np.allclose(query_vector, np.array([0.5] * n_dim))
    assert query.filters == []
    assert query.limit == 5


def test_filter_passes_filter(simple_index):  # noqa: F811
    index = simple_index

    filter = {"number": {"$lt": 1}}
    query = index.build_query().filter(query=filter).build(limit=11)  # type: ignore[attr-defined]

    assert query.vector_fields == {}
    assert query.filters == [{"query": filter}]
    assert query.limit == 11


def test_execute_query_find_filter(simple_index_with_docs, n_dim):  # noqa: F811
    """Tests filters passed to vector search behave as expected"""
    index, _ = simple_index_with_docs

    find_query = np.ones(n_dim)
    filter_query1 = {"number": {"$lt": 8}}
    filter_query2 = {"number": {"$gt": 5}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .filter(query=filter_query1)
        .filter(query=filter_query2)
        .build(limit=5)
    )

    def trial():
        res = index.execute_query(query)
        assert len(res.documents) == 2
        assert set(res.documents.number) == {6, 7}

    assert_when_ready(trial)


def test_execute_only_filter(
    simple_index_with_docs,  # noqa: F811
):
    index, _ = simple_index_with_docs

    filter_query1 = {"number": {"$lt": 8}}
    filter_query2 = {"number": {"$gt": 5}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .filter(query=filter_query1)
        .filter(query=filter_query2)
        .build(limit=5)
    )

    def trial():
        res = index.execute_query(query)

        assert len(res.documents) == 2
        assert set(res.documents.number) == {6, 7}

    assert_when_ready(trial)


def test_execute_text_search_with_filter(
    simple_index_with_docs,  # noqa: F811
):
    """Note: Text search returns only matching _, not limit."""
    index, _ = simple_index_with_docs

    filter_query1 = {"number": {"$eq": 0}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .text_search(query="Python is a valuable skill", search_field='text')
        .filter(query=filter_query1)
        .build(limit=5)
    )

    def trial():
        res = index.execute_query(query)

        assert len(res.documents) == 1
        assert set(res.documents.number) == {0}

    assert_when_ready(trial)


def test_find(
    simple_index_with_docs, n_dim,  # noqa: F811
):
    index, _ = simple_index_with_docs
    limit = 3
    # Base Case: No filters, single text search, single vector search
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=np.ones(n_dim), search_field='embedding')
        .build(limit=limit)
    )

    def trial():
        res = index.execute_query(query)
        assert len(res.documents) == limit
        assert res.documents.number == [5, 4, 6]
    assert_when_ready(trial)


def test_hybrid_search(
    simple_index_with_docs, n_dim  # noqa: F811
):
    find_query = np.ones(n_dim)
    index, docs = simple_index_with_docs
    n_docs = len(docs)
    limit = n_docs

    # Base Case: No filters, single text search, single vector search
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        .build(limit=limit)
    )

    def trial():
        res = index.execute_query(query)
        assert len(res.documents) == limit
        assert set(res.documents.number) == set(range(n_docs))
    assert_when_ready(trial)

    # Now that we've successfully executed a query, we know that the search indexes have been built
    # We no longer need to sleep and retry. Re-run to keep results
    res_base = index.execute_query(query)

    # Case 2: Base plus a filter
    filter_query1 = {"number": {"$gt": 0}}

    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        .filter(query=filter_query1)
        .build(limit=n_docs)
    )

    res = index.execute_query(query)
    assert len(res.documents) == 9
    assert set(res.documents.number) == set(range(1, n_docs))

    # Case 3: Base with, but matching, additional vector search component
    # As we are using averaging to combine embedding vectors, this is a no-op
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .find(query=find_query, search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        .build(limit=n_docs)
    )
    res3 = index.execute_query(query)
    assert res3.documents.number == res_base.documents.number

    # Case 4: Base with, but perpendicular, additional vector search component
    query = (
        index.build_query()  # type: ignore[attr-defined]
        # .find(query=find_query, search_field='embedding')
        .find(query=np.random.standard_normal(find_query.shape), search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        .build(limit=n_docs)
    )
    res4 = index.execute_query(query)
    assert res4.documents.number != res_base.documents.number

    # Case 5: Multiple text searches
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .text_search(query="Python is a valuable skill", search_field='text')
        .text_search(query="classical music compositions", search_field='text')
        .build(limit=n_docs)
    )
    res5 = index.execute_query(query)
    assert res5.documents.number[:2] == [0, 3]

    # Case 6: Multiple text search with filters
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=find_query, search_field='embedding')
        .filter(query={"number": {"$gt": 0}})
        .text_search(query="classical music compositions", search_field='text')
        .text_search(query="Python is a valuable skill", search_field='text')
        .build(limit=n_docs)
    )
    res6 = index.execute_query(query)
    assert res6.documents.number[0] == 3


def test_hybrid_search_multiple_text(
    simple_index_with_docs, n_dim  # noqa: F811
):
    """Tests disambiguation of scores on multiple text searches on same field."""

    index, _ = simple_index_with_docs
    limit = 10
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .text_search(query="classical music compositions", search_field='text')
        .text_search(query="Python is a valuable skill", search_field='text')
        .find(query=np.ones(n_dim), search_field='embedding')
        .build(limit=limit)
    )

    def trial():
        res = index.execute_query(query, score_breakdown=True)
        assert len(res.documents) == limit
        assert res.documents.number == [0, 3, 5, 4, 6, 9, 7, 1, 2, 8]

    assert_when_ready(trial)


def test_hybrid_search_only_text(
    simple_index_with_docs  # noqa: F811
):
    """Query built with two text searches will be a Hybrid Search.

     It will return only two results.
     In our case, each text matches just one document, hence we will receive two results, each top ranked
     """
    index, _ = simple_index_with_docs
    limit = 10
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .text_search(query="classical music compositions", search_field='text')
        .text_search(query="Python is a valuable skill", search_field='text')
        .build(limit=limit)
    )

    def trial():
        res = index.execute_query(query)
        assert len(res.documents) != limit
        # Instead, we find the number of documents containing one of these phrases
        assert len(res.documents) == len(query.text_searches)
        assert set(res.documents.number) == {0, 3}
        assert set(res.scores) == {0.5, 0.5}

    assert_when_ready(trial)


def test_hybrid_search_only_vector(
    simple_index_with_docs, n_dim  # noqa: F811
):

    limit = 3
    index, _ = simple_index_with_docs
    query = (
        index.build_query()  # type: ignore[attr-defined]
        .find(query=np.ones(n_dim), search_field='embedding')
        .find(query=np.zeros(n_dim), search_field='embedding')
        .build(limit=limit)
    )

    def trial():
        res = index.execute_query(query)
        assert len(res.documents) == limit
        assert res.documents.number == [5, 4, 6]
    assert_when_ready(trial)


@pytest.mark.skip
def test_hybrid_search_vectors_with_different_fields(mongodb_index_config):  # noqa: F811
    """Hybrid Search involving queries to two different vector indexes.

    # TODO - To be added in an upcoming release.
    """

    from docarray.index.backends.mongodb_atlas import MongoDBAtlasDocumentIndex
    from tests.index.mongo_atlas import FlatSchema
    multi_index = MongoDBAtlasDocumentIndex[FlatSchema](**mongodb_index_config)
    multi_index._collection.delete_many({})

    n_dim = 25
    n_docs = 5
    data = [FlatSchema(embedding1=np.random.standard_normal(n_dim),
                       embedding2=np.random.standard_normal(n_dim)) for _ in range(n_docs)]
    multi_index.index(data)
    yield multi_index
    multi_index._collection.delete_many({})


    limit = 3
    query = (
        flat_multiple_index.build_query()  # type: ignore[attr-defined]
        .find(query=np.ones(n_dim), search_field='embedding1')
        .find(query=np.zeros(n_dim), search_field='embedding2')
        .build(limit=limit)
    )

    with pytest.raises(NotImplementedError):
        def trial():
            res = multi_index.execute_query(query)
            assert len(res.documents) == limit
            assert res.documents.number == [5, 4, 6]
        assert_when_ready(trial)