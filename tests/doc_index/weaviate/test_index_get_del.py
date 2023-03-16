import logging

import numpy as np
import pytest
import weaviate
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.weaviate_doc_index import WeaviateDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000, is_embedding=True)


class Document(BaseDocument):
    embedding: NdArray[2] = Field(dim=2, is_embedding=True)
    text: str = Field()


@pytest.fixture
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


@pytest.fixture
def weaviate_client():
    client = weaviate.Client("http://weaviate:8080")
    client.schema.delete_all()
    yield client
    client.schema.delete_all()


@pytest.fixture
def documents():

    texts = ["lorem ipsum", "dolor sit amet", "consectetur adipiscing elit"]
    embeddings = [[10, 10], [10.5, 10.5], [-100, -100]]

    # create the docs by enumerating from 1 and use that as the id
    docs = [
        Document(id=i, embedding=embedding, text=text)
        for i, (embedding, text) in enumerate(zip(embeddings, texts))
    ]

    yield docs


@pytest.fixture
def test_store(weaviate_client, documents):
    store = WeaviateDocumentIndex[Document]()
    store.index(documents)
    yield store


def test_index_simple_schema(weaviate_client, ten_simple_docs):
    store = WeaviateDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)
    assert store.num_docs() == 10

    for doc in ten_simple_docs:
        doc_id = doc.id
        doc_embedding = doc.tens

        result = (
            weaviate_client.query.get("Document", "__id")
            .with_additional("vector")
            .with_where({"path": ["__id"], "operator": "Equal", "valueString": doc_id})
            .do()
        )

        result = result["data"]["Get"]["Document"][0]
        assert result["__id"] == doc_id
        assert np.allclose(result["_additional"]["vector"], doc_embedding)


def test_validate_columns(weaviate_client):
    dbconfig = WeaviateDocumentIndex.DBConfig(host="http://weaviate:8080")

    class InvalidDoc1(BaseDocument):
        tens: NdArray[10] = Field(dim=1000, is_embedding=True)
        tens2: NdArray[10] = Field(dim=1000, is_embedding=True)

    class InvalidDoc2(BaseDocument):
        tens: int = Field(dim=1000, is_embedding=True)

    with pytest.raises(ValueError, match=r"Only one column can be marked as embedding"):
        WeaviateDocumentIndex[InvalidDoc1](db_config=dbconfig)

    with pytest.raises(ValueError, match=r"marked as embedding but is not of type"):
        WeaviateDocumentIndex[InvalidDoc2](db_config=dbconfig)


def test_find(weaviate_client, caplog):
    class Document(BaseDocument):
        embedding: NdArray[2] = Field(dim=2, is_embedding=True)

    vectors = [[10, 10], [10.5, 10.5], [-100, -100]]
    docs = [Document(embedding=vector) for vector in vectors]

    store = WeaviateDocumentIndex[Document]()
    store.index(docs)

    query = [10.1, 10.1]

    results = store.find(query, search_field=None, limit=3, distance=1e-2)
    assert len(results) == 2

    results = store.find(query, search_field=None, limit=3, certainty=0.99)
    assert len(results) == 2

    with pytest.raises(
        ValueError,
        match=r"Cannot have both 'certainty' and 'distance' at the same time",
    ):
        store.find(query, search_field=None, limit=3, certainty=0.99, distance=1e-2)

    with caplog.at_level(logging.DEBUG):
        store.find(query, search_field="foo", limit=10)
        assert (
            "Argument search_field is not supported for WeaviateDocumentIndex"
            in caplog.text
        )


def test_find_batched(weaviate_client, caplog):
    class Document(BaseDocument):
        embedding: NdArray[2] = Field(dim=2, is_embedding=True)

    vectors = [[10, 10], [10.5, 10.5], [-100, -100]]
    docs = [Document(embedding=vector) for vector in vectors]

    store = WeaviateDocumentIndex[Document]()
    store.index(docs)

    queries = [[10.1, 10.1], [-100, -100]]

    results = store.find_batched(queries, search_field=None, limit=3, distance=1e-2)
    assert len(results) == 1
    assert len(results[0]) == 2

    results = store.find_batched(queries, search_field=None, limit=3, certainty=0.99)
    assert len(results) == 1
    assert len(results[0]) == 2

    with pytest.raises(
        ValueError,
        match=r"Cannot have both 'certainty' and 'distance' at the same time",
    ):
        store.find_batched(
            queries, search_field=None, limit=3, certainty=0.99, distance=1e-2
        )

    with caplog.at_level(logging.DEBUG):
        store.find_batched(queries, search_field="foo", limit=10)
        assert (
            "Argument search_field is not supported for WeaviateDocumentIndex"
            in caplog.text
        )


@pytest.mark.parametrize(
    "filter_query, expected_num_docs",
    [
        ({"path": ["text"], "operator": "Equal", "valueText": "lorem ipsum"}, 1),
        ({"path": ["text"], "operator": "Equal", "valueText": "foo"}, 0),
    ],
)
def test_filter(test_store, filter_query, expected_num_docs):

    docs = test_store.filter(filter_query, limit=3)
    actual_num_docs = len(docs)

    assert actual_num_docs == expected_num_docs
