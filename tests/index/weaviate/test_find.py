import numpy as np
import pytest
import torch
import weaviate
from pydantic import Field

from docarray import BaseDoc
from docarray.index.backends.weaviate import WeaviateDocumentIndex
from docarray.typing import TorchTensor


@pytest.fixture
def weaviate_client():
    client = weaviate.Client("http://weaviate:8080")
    client.schema.delete_all()
    yield client
    client.schema.delete_all()


def test_find_torch(weaviate_client):
    class TorchDoc(BaseDoc):
        tens: TorchTensor[10] = Field(dims=10, is_embedding=True)

    store = WeaviateDocumentIndex[TorchDoc]()

    index_docs = [
        TorchDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    store.index(index_docs)

    query = index_docs[-1]
    docs, scores = store.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TorchTensor)

    assert docs[0].id == index_docs[-1].id
    assert torch.allclose(docs[0].tens, index_docs[-1].tens)
