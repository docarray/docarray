import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray

import qdrant_client

pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


class NestedDoc(BaseDoc):
    d: SimpleDoc
    tens: NdArray[50]


@pytest.fixture
def qdrant_config():
    return QdrantDocumentIndex.DBConfig()


@pytest.fixture
def qdrant():
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient('http://localhost:6333')
    client.delete_collection(collection_name='documents')


def test_persist_and_restore(qdrant_config, qdrant):
    query = SimpleDoc(tens=np.random.random((10,)))

    # create index
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    store.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert store.num_docs() == 10
    find_results_before = store.find(query, search_field='tens', limit=5)

    # delete and restore
    del store
    store = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    assert store.num_docs() == 10
    find_results_after = store.find(query, search_field='tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert (doc_before.tens == doc_after.tens).all()

    # add new data
    store.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(5)])
    assert store.num_docs() == 15


def test_persist_and_restore_nested(qdrant_config, qdrant):
    query = NestedDoc(
        tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
    )

    # create index
    store = QdrantDocumentIndex[NestedDoc](db_config=qdrant_config)
    store.index(
        [
            NestedDoc(
                tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
            )
            for _ in range(10)
        ]
    )
    assert store.num_docs() == 10
    find_results_before = store.find(query, search_field='d__tens', limit=5)

    # delete and restore
    del store
    store = QdrantDocumentIndex[NestedDoc](db_config=qdrant_config)
    assert store.num_docs() == 10
    find_results_after = store.find(query, search_field='d__tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert (doc_before.tens == doc_after.tens).all()

    # delete and restore
    store.index(
        [
            NestedDoc(
                tens=np.random.random((50,)), d=SimpleDoc(tens=np.random.random((10,)))
            )
            for _ in range(5)
        ]
    )
    assert store.num_docs() == 15
