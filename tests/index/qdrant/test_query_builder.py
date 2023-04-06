import pytest
import qdrant_client
import numpy as np

from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray

from qdrant_client.http import models as rest


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]
    number: int
    text: str


class SimpleSchema(BaseDoc):
    embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
    number: int
    text: str


@pytest.fixture
def qdrant_config():
    return QdrantDocumentIndex.DBConfig()


@pytest.fixture
def qdrant():
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient('http://localhost:6333')
    client.delete_collection(collection_name='documents')


def test_find_uses_provided_vector(qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    query = store.build_query().find(np.ones(10), 'embedding').build(7)  # type: ignore[attr-defined]

    assert query.vector_field == 'embedding'
    assert np.allclose(query.vector_query, np.ones(10))
    assert query.filter is None
    assert query.limit == 7


def test_multiple_find_returns_averaged_vector(qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    query = (
        store.build_query()  # type: ignore[attr-defined]
        .find(np.ones(10), 'embedding')
        .find(np.zeros(10), 'embedding')
        .build(5)
    )

    assert query.vector_field == 'embedding'
    assert np.allclose(query.vector_query, np.array([0.5] * 10))
    assert query.filter is None
    assert query.limit == 5


def test_multiple_find_different_field_raises_error(qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    with pytest.raises(ValueError):
        (
            store.build_query()  # type: ignore[attr-defined]
            .find(np.ones(10), 'embedding_1')
            .find(np.zeros(10), 'embedding_2')
        )


def test_filter_passes_qdrant_filter(qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    qdrant_filter = rest.Filter(should=[rest.HasIdCondition(has_id=[1, 2, 3])])
    query = store.build_query().filter(qdrant_filter).build(11)  # type: ignore[attr-defined]

    assert query.vector_field is None
    assert query.vector_query is None
    assert query.filter == rest.Filter(must=[qdrant_filter])
    assert query.limit == 11


def test_text_search_creates_qdrant_filter(qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    query = store.build_query().text_search('lorem ipsum', 'text').build(3)  # type: ignore[attr-defined]

    assert query.vector_field is None
    assert query.vector_query is None
    assert isinstance(query.filter, rest.Filter)
    assert len(query.filter.must) == 1  # type: ignore[arg-type]
    assert isinstance(query.filter.must[0], rest.FieldCondition)  # type: ignore[index]
    assert query.filter.must[0].key == 'text'  # type: ignore[index]
    assert query.filter.must[0].match.text == 'lorem ipsum'  # type: ignore[index, union-attr]
    assert query.limit == 3


def test_query_builder_execute_query_find_text_search_filter(qdrant_config, qdrant):
    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.ones(10),
            number=i,
            text=f'Lorem ipsum {i}',
        )
        for i in range(10, 30, 2)
    ]
    store.index(index_docs)

    find_query = np.ones(10)
    text_search_query = 'ipsum 1'
    filter_query = rest.Filter(
        must=[
            rest.FieldCondition(
                key='number',
                range=rest.Range(
                    gte=12,
                    lt=18,
                ),
            )
        ]
    )
    query = (
        store.build_query()  # type: ignore[attr-defined]
        .find(find_query, search_field='embedding')
        .text_search(text_search_query, search_field='text')
        .filter(filter_query)
        .build(limit=5)
    )
    docs = store.execute_query(query)

    assert len(docs) == 3
    assert all(x in docs.number for x in [12, 14, 16])
