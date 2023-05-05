import os

import pytest
import qdrant_client

from docarray.index import QdrantDocumentIndex

cur_dir = os.path.dirname(os.path.abspath(__file__))
qdrant_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture
def qdrant() -> qdrant_client.QdrantClient:
    """This fixture takes care of removing the collection before each test case"""
    client = qdrant_client.QdrantClient(path='/tmp/qdrant-local')
    client.delete_collection(collection_name='documents')
    return client


@pytest.fixture
def qdrant_config(qdrant) -> QdrantDocumentIndex.DBConfig:
    return QdrantDocumentIndex.DBConfig(path=qdrant._client.location)


@pytest.fixture(scope='session', autouse=True)
def start_storage():
    os.system(f"docker-compose -f {qdrant_yml} up -d --remove-orphans")

    yield
    os.system(f"docker-compose -f {qdrant_yml} down --remove-orphans")
