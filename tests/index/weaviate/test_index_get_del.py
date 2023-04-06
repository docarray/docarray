import logging

import numpy as np
import pytest
import weaviate
from pydantic import Field

from docarray import BaseDoc
from docarray.documents import ImageDoc
from docarray.index.backends.weaviate import DOCUMENTID, WeaviateDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000, is_embedding=True)


class Document(BaseDoc):
    embedding: NdArray[2] = Field(dim=2, is_embedding=True)
    text: str = Field()


class NestedDocument(BaseDoc):
    text: str = Field()
    child: Document


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
        Document(id=str(i), embedding=embedding, text=text)
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
            weaviate_client.query.get("Document", DOCUMENTID)
            .with_additional("vector")
            .with_where(
                {"path": [DOCUMENTID], "operator": "Equal", "valueString": doc_id}
            )
            .do()
        )

        result = result["data"]["Get"]["Document"][0]
        assert result[DOCUMENTID] == doc_id
        assert np.allclose(result["_additional"]["vector"], doc_embedding)


def test_validate_columns(weaviate_client):
    dbconfig = WeaviateDocumentIndex.DBConfig(host="http://weaviate:8080")

    class InvalidDoc1(BaseDoc):
        tens: NdArray[10] = Field(dim=1000, is_embedding=True)
        tens2: NdArray[10] = Field(dim=1000, is_embedding=True)

    class InvalidDoc2(BaseDoc):
        tens: int = Field(dim=1000, is_embedding=True)

    with pytest.raises(ValueError, match=r"Only one column can be marked as embedding"):
        WeaviateDocumentIndex[InvalidDoc1](db_config=dbconfig)

    with pytest.raises(ValueError, match=r"marked as embedding but is not of type"):
        WeaviateDocumentIndex[InvalidDoc2](db_config=dbconfig)


def test_find(weaviate_client, caplog):
    class Document(BaseDoc):
        embedding: NdArray[2] = Field(dim=2, is_embedding=True)

    vectors = [[10, 10], [10.5, 10.5], [-100, -100]]
    docs = [Document(embedding=vector) for vector in vectors]

    store = WeaviateDocumentIndex[Document]()
    store.index(docs)

    query = [10.1, 10.1]

    results = store.find(
        query, search_field='', limit=3, score_name="distance", score_threshold=1e-2
    )
    assert len(results) == 2

    results = store.find(query, search_field='', limit=3, score_threshold=0.99)
    assert len(results) == 2

    with pytest.raises(
        ValueError,
        match=r"Argument search_field is not supported for WeaviateDocumentIndex",
    ):
        store.find(query, search_field="foo", limit=10)


def test_find_batched(weaviate_client, caplog):
    class Document(BaseDoc):
        embedding: NdArray[2] = Field(dim=2, is_embedding=True)

    vectors = [[10, 10], [10.5, 10.5], [-100, -100]]
    docs = [Document(embedding=vector) for vector in vectors]

    store = WeaviateDocumentIndex[Document]()
    store.index(docs)

    queries = np.array([[10.1, 10.1], [-100, -100]])

    results = store.find_batched(
        queries, search_field='', limit=3, score_name="distance", score_threshold=1e-2
    )
    assert len(results) == 2
    assert len(results.documents[0]) == 2
    assert len(results.documents[1]) == 1

    results = store.find_batched(
        queries, search_field='', limit=3, score_name="certainty"
    )
    assert len(results) == 2
    assert len(results.documents[0]) == 3
    assert len(results.documents[1]) == 3

    with pytest.raises(
        ValueError,
        match=r"Argument search_field is not supported for WeaviateDocumentIndex",
    ):
        store.find_batched(queries, search_field="foo", limit=10)


@pytest.mark.parametrize(
    "filter_query, expected_num_docs",
    [
        ({"path": ["text"], "operator": "Equal", "valueText": "lorem ipsum"}, 1),
        ({"path": ["text"], "operator": "Equal", "valueText": "foo"}, 0),
        ({"path": ["id"], "operator": "Equal", "valueString": "1"}, 1),
    ],
)
def test_filter(test_store, filter_query, expected_num_docs):
    docs = test_store.filter(filter_query, limit=3)
    actual_num_docs = len(docs)

    assert actual_num_docs == expected_num_docs


@pytest.mark.parametrize(
    "filter_queries, expected_num_docs",
    [
        (
            [
                {"path": ["text"], "operator": "Equal", "valueText": "lorem ipsum"},
                {"path": ["text"], "operator": "Equal", "valueText": "foo"},
            ],
            [1, 0],
        ),
        (
            [
                {"path": ["id"], "operator": "Equal", "valueString": "1"},
                {"path": ["id"], "operator": "Equal", "valueString": "2"},
            ],
            [1, 0],
        ),
    ],
)
def test_filter_batched(test_store, filter_queries, expected_num_docs):
    filter_queries = [
        {"path": ["text"], "operator": "Equal", "valueText": "lorem ipsum"},
        {"path": ["text"], "operator": "Equal", "valueText": "foo"},
    ]

    results = test_store.filter_batched(filter_queries, limit=3)
    actual_num_docs = [len(docs) for docs in results]
    assert actual_num_docs == expected_num_docs


def test_text_search(test_store):
    results = test_store.text_search(query="lorem", search_field="text", limit=3)
    assert len(results.documents) == 1


def test_text_search_batched(test_store):
    text_queries = ["lorem", "foo"]

    results = test_store.text_search_batched(
        queries=text_queries, search_field="text", limit=3
    )
    assert len(results.documents[0]) == 1
    assert len(results.documents[1]) == 0


def test_del_items(test_store):
    del test_store[["1", "2"]]
    assert test_store.num_docs() == 1


def test_get_items(test_store):
    docs = test_store[["1", "2"]]
    assert len(docs) == 2
    assert set(doc.id for doc in docs) == {'1', '2'}


def test_index_nested_documents(weaviate_client):
    store = WeaviateDocumentIndex[NestedDocument]()
    document = NestedDocument(
        text="lorem ipsum", child=Document(embedding=[10, 10], text="dolor sit amet")
    )
    store.index([document])
    assert store.num_docs() == 1


@pytest.mark.parametrize(
    "search_field, query, expected_num_docs",
    [
        ("text", "lorem", 1),
        ("child__text", "dolor", 1),
        ("text", "foo", 0),
        ("child__text", "bar", 0),
    ],
)
def test_text_search_nested_documents(
    weaviate_client, search_field, query, expected_num_docs
):
    store = WeaviateDocumentIndex[NestedDocument]()
    document = NestedDocument(
        text="lorem ipsum", child=Document(embedding=[10, 10], text="dolor sit amet")
    )
    store.index([document])

    results = store.text_search(query=query, search_field=search_field, limit=3)

    assert len(results.documents) == expected_num_docs


def test_reuse_existing_schema(weaviate_client, caplog):
    WeaviateDocumentIndex[SimpleDoc]()

    with caplog.at_level(logging.DEBUG):
        WeaviateDocumentIndex[SimpleDoc]()
        assert "Will reuse existing schema" in caplog.text


def test_query_builder(test_store):
    query_embedding = [10.25, 10.25]
    query_text = "ipsum"
    where_filter = {"path": ["id"], "operator": "Equal", "valueString": "1"}
    q = (
        test_store.build_query()
        .find(query=query_embedding)
        .filter(where_filter)
        .build()
    )

    docs = test_store.execute_query(q)
    assert len(docs) == 1

    q = (
        test_store.build_query()
        .text_search(query=query_text, search_field="text")
        .build()
    )

    docs = test_store.execute_query(q)
    assert len(docs) == 1


def test_batched_query_builder(test_store):
    query_embeddings = [[10.25, 10.25], [-100, -100]]
    query_texts = ["ipsum", "foo"]
    where_filters = [{"path": ["id"], "operator": "Equal", "valueString": "1"}]

    q = (
        test_store.build_query()
        .find_batched(
            queries=query_embeddings, score_name="certainty", score_threshold=0.99
        )
        .filter_batched(filters=where_filters)
        .build()
    )

    docs = test_store.execute_query(q)
    assert len(docs[0]) == 1
    assert len(docs[1]) == 0

    q = (
        test_store.build_query()
        .text_search_batched(queries=query_texts, search_field="text")
        .build()
    )

    docs = test_store.execute_query(q)
    assert len(docs[0]) == 1
    assert len(docs[1]) == 0


def test_raw_graphql(test_store):
    graphql_query = """
    {
     Aggregate {
      Document {
       meta {
        count
       }
      }
     }
    }
    """

    results = test_store.execute_query(graphql_query)
    num_docs = results["data"]["Aggregate"]["Document"][0]["meta"]["count"]

    assert num_docs == 3


def test_hybrid_query(test_store):
    query_embedding = [10.25, 10.25]
    query_text = "ipsum"
    where_filter = {"path": ["id"], "operator": "Equal", "valueString": "1"}

    q = (
        test_store.build_query()
        .find(query=query_embedding)
        .text_search(query=query_text, search_field="text")
        .filter(where_filter)
        .build()
    )

    docs = test_store.execute_query(q)
    assert len(docs) == 1


def test_hybrid_query_batched(test_store):
    query_embeddings = [[10.25, 10.25], [-100, -100]]
    query_texts = ["dolor", "elit"]

    q = (
        test_store.build_query()
        .find_batched(
            queries=query_embeddings, score_name="certainty", score_threshold=0.99
        )
        .text_search_batched(queries=query_texts, search_field="text")
        .build()
    )

    docs = test_store.execute_query(q)
    assert docs[0][0].id == '1'
    assert docs[1][0].id == '2'


def test_index_document_with_bytes(weaviate_client):
    doc = ImageDoc(id="1", url="www.foo.com", bytes_=b"foo")

    store = WeaviateDocumentIndex[ImageDoc]()
    store.index([doc])

    results = store.filter(
        filter_query={"path": ["id"], "operator": "Equal", "valueString": "1"}
    )

    assert doc == results[0]


def test_index_document_with_no_embeddings(weaviate_client):
    # define a document that does not have any field where is_embedding=True
    class Document(BaseDoc):
        not_embedding: NdArray[2] = Field(dim=2)
        text: str

    doc = Document(not_embedding=[2, 5], text="dolor sit amet", id="1")

    store = WeaviateDocumentIndex[Document]()

    store.index([doc])

    results = store.filter(
        filter_query={"path": ["id"], "operator": "Equal", "valueString": "1"}
    )

    assert doc == results[0]
