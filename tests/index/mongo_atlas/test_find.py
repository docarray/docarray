import numpy as np
from pydantic import Field

from docarray import BaseDoc
from docarray.index import MongoAtlasDocumentIndex
from docarray.typing import NdArray

from .fixtures import *  # noqa
from .helpers import assert_when_ready

N_DIM = 10


def test_find_simple_schema(simple_index_with_docs, simple_schema):

    simple_index, random_simple_documents = simple_index_with_docs
    query = np.ones(N_DIM)
    closest_document = simple_schema(embedding=query, text="other", number=10)
    simple_index.index(closest_document)

    def pred():
        docs, scores = simple_index.find(query, search_field='embedding', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding, closest_document.embedding)

    assert_when_ready(pred)


def test_find_empty_index(simple_index):
    query = np.random.rand(N_DIM)

    def pred():
        docs, scores = simple_index.find(query, search_field='embedding', limit=5)
        assert len(docs) == 0
        assert len(scores) == 0

    assert_when_ready(pred)


def test_find_limit_larger_than_index(simple_index_with_docs, simple_schema):
    simple_index, random_simple_documents = simple_index_with_docs

    query = np.ones(N_DIM)
    new_doc = simple_schema(embedding=query, text="other", number=10)

    simple_index.index(new_doc)

    def pred():
        docs, scores = simple_index.find(query, search_field='embedding', limit=20)
        assert len(docs) == 11
        assert len(scores) == 11

    assert_when_ready(pred)


def test_find_flat_schema(mongo_fixture_env, clean_database):
    class FlatSchema(BaseDoc):
        embedding1: NdArray = Field(dim=N_DIM, index_name="vector_index_1")
        # the dim and N_DIM are setted different on propouse. to check the correct handling of n_dim
        embedding2: NdArray[50] = Field(dim=N_DIM, index_name="vector_index_2")

    uri, database_name, collection_name = mongo_fixture_env
    index = MongoAtlasDocumentIndex[FlatSchema](
        mongo_connection_uri=uri,
        database_name=database_name,
        collection_name=collection_name,
    )

    index_docs = [
        FlatSchema(embedding1=np.random.rand(N_DIM), embedding2=np.random.rand(50))
        for _ in range(10)
    ]

    index_docs.append(FlatSchema(embedding1=np.zeros(N_DIM), embedding2=np.ones(50)))
    index_docs.append(FlatSchema(embedding1=np.ones(N_DIM), embedding2=np.zeros(50)))
    index.index(index_docs)

    query = (np.ones(N_DIM), np.ones(50))

    def pred1():

        # find on embedding1
        docs, scores = index.find(query[0], search_field='embedding1', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding1, index_docs[-1].embedding1)
        assert np.allclose(docs[0].embedding2, index_docs[-1].embedding2)

    assert_when_ready(pred1)

    def pred2():
        # find on embedding2
        docs, scores = index.find(query[1], search_field='embedding2', limit=5)
        assert len(docs) == 5
        assert len(scores) == 5
        assert np.allclose(docs[0].embedding1, index_docs[-2].embedding1)
        assert np.allclose(docs[0].embedding2, index_docs[-2].embedding2)

    assert_when_ready(pred2)


def test_find_batches(simple_index_with_docs):

    simple_index, docs = simple_index_with_docs
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


def test_find_nested_schema(nested_index_with_docs, nested_schema):
    db, base_docs = nested_index_with_docs

    query = nested_schema[0](
        d=nested_schema[1](embedding=np.ones(N_DIM)), embedding=np.ones(N_DIM)
    )

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
