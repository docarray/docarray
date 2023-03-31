import pytest
import qdrant_client
import numpy as np

from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)
    text: str


@pytest.fixture
def qdrant_config():
    return QdrantDocumentIndex.DBConfig()


@pytest.fixture
def qdrant():
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient('http://localhost:6333')
    client.delete_collection(collection_name='documents')


def test_text_search(qdrant_config, qdrant):
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')
        text: str

    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            text=f'Lorem ipsum {i}',
        )
        for i in range(10)
    ]
    store.index(index_docs)

    query = 'ipsum 2'
    docs, scores = store.text_search(query, search_field='text', limit=5)

    assert len(docs) == 1
    assert len(scores) >= 1  # TODO: that should be == 1
    assert docs[0].id == index_docs[2].id
    assert scores[0] > 0.0
