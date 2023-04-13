import pytest
import qdrant_client

from docarray.index import QdrantDocumentIndex


@pytest.fixture
def qdrant_config():
    return QdrantDocumentIndex.DBConfig()


@pytest.fixture
def qdrant():
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient('http://localhost:6333')
    client.delete_collection(collection_name='documents')
