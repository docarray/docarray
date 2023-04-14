# TODO: enable ruff qa on this file when we figure out why it thinks weaviate_client is
#       redefined at each test that fixture
# ruff: noqa
import numpy as np
import pytest
import torch
from pydantic import Field

from docarray import BaseDoc
from docarray.index.backends.weaviate import WeaviateDocumentIndex
from docarray.typing import TorchTensor
from tests.index.weaviate.fixture_weaviate import (  # noqa: F401
    start_storage,
    weaviate_client,
)

pytestmark = [pytest.mark.slow, pytest.mark.index]


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


@pytest.mark.tensorflow
def test_find_tensorflow():
    from docarray.typing import TensorFlowTensor

    class TfDoc(BaseDoc):
        tens: TensorFlowTensor[10] = Field(dims=10, is_embedding=True)

    store = WeaviateDocumentIndex[TfDoc]()

    index_docs = [
        TfDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    store.index(index_docs)

    query = index_docs[-1]
    docs, scores = store.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(
        docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )
