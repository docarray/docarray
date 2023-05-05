# TODO: enable ruff qa on this file when we figure out why it thinks weaviate_client is
#       redefined at each test that fixture
# ruff: noqa
import logging

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.documents import ImageDoc, TextDoc
from docarray.index.backends.weaviate import (
    DOCUMENTID,
    EmbeddedOptions,
    WeaviateDocumentIndex,
)
from docarray.typing import NdArray
from tests.index.weaviate.fixture_weaviate import (  # noqa: F401
    HOST,
    start_storage,
    weaviate_client,
)

pytestmark = [pytest.mark.slow, pytest.mark.index]


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
def test_index(weaviate_client, documents):
    index = WeaviateDocumentIndex[Document]()
    index.index(documents)
    yield index


def test_index_simple_schema(weaviate_client, ten_simple_docs):
    index = WeaviateDocumentIndex[SimpleDoc](index_name="Document")
    index.index(ten_simple_docs)
    assert index.num_docs() == 10

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
    dbconfig = WeaviateDocumentIndex.DBConfig(host=HOST)

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

    index = WeaviateDocumentIndex[Document]()
    index.index(docs)

    query = [10.1, 10.1]

    results = index.find(
        query, search_field='', limit=3, score_name="distance", score_threshold=1e-2
    )
    assert len(results) == 2

    results = index.find(query, search_field='', limit=3, score_threshold=0.99)
    assert len(results) == 2

    with pytest.raises(
        ValueError,
        match=r"Argument search_field is not supported for WeaviateDocumentIndex",
    ):
        index.find(query, search_field="foo", limit=10)


def test_find_batched(weaviate_client, caplog):
    class Document(BaseDoc):
        embedding: NdArray[2] = Field(dim=2, is_embedding=True)

    vectors = [[10, 10], [10.5, 10.5], [-100, -100]]
    docs = [Document(embedding=vector) for vector in vectors]

    index = WeaviateDocumentIndex[Document]()
    index.index(docs)

    queries = np.array([[10.1, 10.1], [-100, -100]])

    results = index.find_batched(
        queries, search_field='', limit=3, score_name="distance", score_threshold=1e-2
    )
    assert len(results) == 2
    assert len(results.documents[0]) == 2
    assert len(results.documents[1]) == 1

    results = index.find_batched(
        queries, search_field='', limit=3, score_name="certainty"
    )
    assert len(results) == 2
    assert len(results.documents[0]) == 3
    assert len(results.documents[1]) == 3

    with pytest.raises(
        ValueError,
        match=r"Argument search_field is not supported for WeaviateDocumentIndex",
    ):
        index.find_batched(queries, search_field="foo", limit=10)


@pytest.mark.parametrize(
    "filter_query, expected_num_docs",
    [
        ({"path": ["text"], "operator": "Equal", "valueText": "lorem ipsum"}, 1),
        ({"path": ["text"], "operator": "Equal", "valueText": "foo"}, 0),
        ({"path": ["id"], "operator": "Equal", "valueString": "1"}, 1),
    ],
)
def test_filter(test_index, filter_query, expected_num_docs):
    docs = test_index.filter(filter_query, limit=3)
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
def test_filter_batched(test_index, filter_queries, expected_num_docs):
    filter_queries = [
        {"path": ["text"], "operator": "Equal", "valueText": "lorem ipsum"},
        {"path": ["text"], "operator": "Equal", "valueText": "foo"},
    ]

    results = test_index.filter_batched(filter_queries, limit=3)
    actual_num_docs = [len(docs) for docs in results]
    assert actual_num_docs == expected_num_docs


def test_text_search(test_index):
    results = test_index.text_search(query="lorem", search_field="text", limit=3)
    assert len(results.documents) == 1


def test_text_search_batched(test_index):
    text_queries = ["lorem", "foo"]

    results = test_index.text_search_batched(
        queries=text_queries, search_field="text", limit=3
    )
    assert len(results.documents[0]) == 1
    assert len(results.documents[1]) == 0


def test_del_items(test_index):
    del test_index[["1", "2"]]
    assert test_index.num_docs() == 1


def test_get_items(test_index):
    docs = test_index[["1", "2"]]
    assert len(docs) == 2
    assert set(doc.id for doc in docs) == {'1', '2'}


def test_index_nested_documents(weaviate_client):
    index = WeaviateDocumentIndex[NestedDocument]()
    document = NestedDocument(
        text="lorem ipsum", child=Document(embedding=[10, 10], text="dolor sit amet")
    )
    index.index([document])
    assert index.num_docs() == 1


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
    index = WeaviateDocumentIndex[NestedDocument]()
    document = NestedDocument(
        text="lorem ipsum", child=Document(embedding=[10, 10], text="dolor sit amet")
    )
    index.index([document])

    results = index.text_search(query=query, search_field=search_field, limit=3)

    assert len(results.documents) == expected_num_docs


def test_reuse_existing_schema(weaviate_client, caplog):
    WeaviateDocumentIndex[SimpleDoc]()

    with caplog.at_level(logging.DEBUG):
        WeaviateDocumentIndex[SimpleDoc]()
        assert "Will reuse existing schema" in caplog.text


def test_query_builder(test_index):
    query_embedding = [10.25, 10.25]
    query_text = "ipsum"
    where_filter = {"path": ["id"], "operator": "Equal", "valueString": "1"}
    q = (
        test_index.build_query()
        .find(query=query_embedding)
        .filter(where_filter)
        .build()
    )

    docs = test_index.execute_query(q)
    assert len(docs) == 1

    q = (
        test_index.build_query()
        .text_search(query=query_text, search_field="text")
        .build()
    )

    docs = test_index.execute_query(q)
    assert len(docs) == 1


def test_batched_query_builder(test_index):
    query_embeddings = [[10.25, 10.25], [-100, -100]]
    query_texts = ["ipsum", "foo"]
    where_filters = [{"path": ["id"], "operator": "Equal", "valueString": "1"}]

    q = (
        test_index.build_query()
        .find_batched(
            queries=query_embeddings, score_name="certainty", score_threshold=0.99
        )
        .filter_batched(filters=where_filters)
        .build()
    )

    docs = test_index.execute_query(q)
    assert len(docs[0]) == 1
    assert len(docs[1]) == 0

    q = (
        test_index.build_query()
        .text_search_batched(queries=query_texts, search_field="text")
        .build()
    )

    docs = test_index.execute_query(q)
    assert len(docs[0]) == 1
    assert len(docs[1]) == 0


def test_raw_graphql(test_index):
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

    results = test_index.execute_query(graphql_query)
    num_docs = results["data"]["Aggregate"]["Document"][0]["meta"]["count"]

    assert num_docs == 3


def test_hybrid_query(test_index):
    query_embedding = [10.25, 10.25]
    query_text = "ipsum"
    where_filter = {"path": ["id"], "operator": "Equal", "valueString": "1"}

    q = (
        test_index.build_query()
        .find(query=query_embedding)
        .text_search(query=query_text, search_field="text")
        .filter(where_filter)
        .build()
    )

    docs = test_index.execute_query(q)
    assert len(docs) == 1


def test_hybrid_query_batched(test_index):
    query_embeddings = [[10.25, 10.25], [-100, -100]]
    query_texts = ["dolor", "elit"]

    q = (
        test_index.build_query()
        .find_batched(
            queries=query_embeddings, score_name="certainty", score_threshold=0.99
        )
        .text_search_batched(queries=query_texts, search_field="text")
        .build()
    )

    docs = test_index.execute_query(q)
    assert docs[0][0].id == '1'
    assert docs[1][0].id == '2'


def test_index_multi_modal_doc():
    class MyMultiModalDoc(BaseDoc):
        image: ImageDoc
        text: TextDoc

    index = WeaviateDocumentIndex[MyMultiModalDoc]()

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


def test_index_document_with_bytes(weaviate_client):
    doc = ImageDoc(id="1", url="www.foo.com", bytes_=b"foo")

    index = WeaviateDocumentIndex[ImageDoc]()
    index.index([doc])

    results = index.filter(
        filter_query={"path": ["id"], "operator": "Equal", "valueString": "1"}
    )

    assert doc == results[0]


def test_index_document_with_no_embeddings(weaviate_client):
    # define a document that does not have any field where is_embedding=True
    class Document(BaseDoc):
        not_embedding: NdArray[2] = Field(dim=2)
        text: str

    doc = Document(not_embedding=[2, 5], text="dolor sit amet", id="1")

    index = WeaviateDocumentIndex[Document]()

    index.index([doc])

    results = index.filter(
        filter_query={"path": ["id"], "operator": "Equal", "valueString": "1"}
    )

    assert doc == results[0]


def test_limit_query_builder(test_index):
    query_vector = [10.25, 10.25]
    q = test_index.build_query().find(query=query_vector).limit(2)

    docs = test_index.execute_query(q)
    assert len(docs) == 2


def test_embedded_weaviate():
    class Document(BaseDoc):
        text: str

    embedded_options = EmbeddedOptions()
    db_config = WeaviateDocumentIndex.DBConfig(embedded_options=embedded_options)
    index = WeaviateDocumentIndex[Document](db_config=db_config)

    assert index._client._connection.embedded_db
