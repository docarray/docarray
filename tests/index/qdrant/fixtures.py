import uuid

import pytest
import qdrant_client

from docarray.index import QdrantDocumentIndex


@pytest.fixture
def qdrant() -> qdrant_client.QdrantClient:
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient(path='/tmp/qdrant-local')
    client.delete_collection(collection_name='documents')
    return client


@pytest.fixture
def qdrant_config(qdrant) -> QdrantDocumentIndex.DBConfig:
    return QdrantDocumentIndex.DBConfig(path=qdrant._client.location)
