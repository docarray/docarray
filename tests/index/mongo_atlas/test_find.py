import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import MongoAtlasDocumentIndex
from docarray.typing import NdArray

from . import NestedDoc, SimpleDoc, SimpleSchema, assert_when_ready

N_DIM = 10


def test_find_simple_schema(simple_index_with_docs):  # noqa: F811

    simple_index, random_simple_documents = simple_index_with_docs  # noqa: F811
    query = np.ones(N_DIM)

    # Insert one doc that identically matches query's embedding
    expected_matching_document = SimpleSchema(embedding=query, text="other", number=10)
    simple_index.index(expected_matching_document)

    def pred():
        docs, scores = simple_index.find(query, search_field='embedding', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding, expected_matching_document.embedding)

    assert_when_ready(pred)


def test_find_empty_index(simple_index):  # noqa: F811
    query = np.random.rand(N_DIM)

    def pred():
        docs, scores = simple_index.find(query, search_field='embedding', limit=5)
        assert len(docs) == 0
        assert len(scores) == 0

    assert_when_ready(pred)


def test_find_limit_larger_than_index(simple_index_with_docs):  # noqa: F811
    simple_index, random_simple_documents = simple_index_with_docs  # noqa: F811

    query = np.ones(N_DIM)
    new_doc = SimpleSchema(embedding=query, text="other", number=10)

    simple_index.index(new_doc)

    def pred():
        docs, scores = simple_index.find(query, search_field='embedding', limit=20)
        assert len(docs) == 11
        assert len(scores) == 11

    assert_when_ready(pred)


def test_find_flat_schema(mongodb_index_config):  # noqa: F811
    class FlatSchema(BaseDoc):
        embedding1: NdArray = Field(dim=N_DIM, index_name="vector_index_1")
        # the dim and N_DIM are setted different on propouse. to check the correct handling of n_dim
        embedding2: NdArray[50] = Field(dim=N_DIM, index_name="vector_index_2")

    index = MongoAtlasDocumentIndex[FlatSchema](**mongodb_index_config)

    index._doc_collection.delete_many({})

    index_docs = [
        FlatSchema(embedding1=np.random.rand(N_DIM), embedding2=np.random.rand(50))
        for _ in range(10)
    ]

    index_docs.append(FlatSchema(embedding1=np.zeros(N_DIM), embedding2=np.ones(50)))
    index_docs.append(FlatSchema(embedding1=np.ones(N_DIM), embedding2=np.zeros(50)))
    index.index(index_docs)

    def pred1():

        # find on embedding1
        query = np.ones(N_DIM)
        docs, scores = index.find(query, search_field='embedding1', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding1, index_docs[-1].embedding1)
        assert np.allclose(docs[0].embedding2, index_docs[-1].embedding2)

    assert_when_ready(pred1)

    def pred2():
        # find on embedding2
        query = np.ones(50)
        docs, scores = index.find(query, search_field='embedding2', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding1, index_docs[-2].embedding1)
        assert np.allclose(docs[0].embedding2, index_docs[-2].embedding2)

    assert_when_ready(pred2)


def test_find_batches(simple_index_with_docs):  # noqa: F811

    simple_index, docs = simple_index_with_docs  # noqa: F811
    queries = np.array([np.random.rand(10) for _ in range(3)])

    def pred():
        resp = simple_index.find_batched(
            queries=queries, search_field='embedding', limit=10
        )
        docs_responses = resp.documents
        assert len(docs_responses) == 3
        for matches in docs_responses:
            assert len(matches) == 10

    assert_when_ready(pred)


def test_find_nested_schema(nested_index_with_docs):  # noqa: F811
    db, base_docs = nested_index_with_docs

    query = NestedDoc(d=SimpleDoc(embedding=np.ones(N_DIM)), embedding=np.ones(N_DIM))

    # find on root level
    def pred():
        docs, scores = db.find(query, search_field='embedding', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding, base_docs[-1].embedding)

        # find on first nesting level
        docs, scores = db.find(query, search_field='d__embedding', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].d.embedding, base_docs[-2].d.embedding)

    assert_when_ready(pred)


def test_find_schema_without_index(mongodb_index_config):  # noqa: F811
    class Schema(BaseDoc):
        vec: NdArray = Field(dim=N_DIM)

    index = MongoAtlasDocumentIndex[Schema](**mongodb_index_config)
    query = np.ones(N_DIM)
    with pytest.raises(ValueError):
        index.find(query, search_field='vec', limit=2)
