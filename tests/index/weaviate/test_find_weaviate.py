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

    index = WeaviateDocumentIndex[TorchDoc]()

    index_docs = [
        TorchDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    index.index(index_docs)

    query = index_docs[-1]
    docs, scores = index.find(query, limit=5)

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

    index = WeaviateDocumentIndex[TfDoc]()

    index_docs = [
        TfDoc(tens=np.random.rand(10).astype(dtype=np.float32)) for _ in range(10)
    ]
    index.index(index_docs)

    query = index_docs[-1]
    docs, scores = index.find(query, limit=5)

    assert len(docs) == 5
    assert len(scores) == 5
    for doc in docs:
        assert isinstance(doc.tens, TensorFlowTensor)

    assert docs[0].id == index_docs[-1].id
    assert np.allclose(
        docs[0].tens.unwrap().numpy(), index_docs[-1].tens.unwrap().numpy()
    )


def test_comprehensive():
    import numpy as np
    from pydantic import Field

    from docarray import BaseDoc
    from docarray.index.backends.weaviate import WeaviateDocumentIndex
    from docarray.typing import NdArray

    class Document(BaseDoc):
        text: str
        embedding: NdArray[2] = Field(
            dims=2, is_embedding=True
        )  # Embedding column -> vector representation of the document
        file: NdArray[100] = Field(dims=100)

    docs = [
        Document(
            text="Hello world",
            embedding=np.array([1, 2]),
            file=np.random.rand(100),
            id="1",
        ),
        Document(
            text="Hello world, how are you?",
            embedding=np.array([3, 4]),
            file=np.random.rand(100),
            id="2",
        ),
        Document(
            text="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut",
            embedding=np.array([5, 6]),
            file=np.random.rand(100),
            id="3",
        ),
    ]

    batch_config = {
        "batch_size": 20,
        "dynamic": False,
        "timeout_retries": 3,
        "num_workers": 1,
    }

    dbconfig = WeaviateDocumentIndex.DBConfig(
        host="https://docarray-test-4mfexsso.weaviate.network",  # Replace with your endpoint
        auth_api_key="JPsfPHB3OLHrgnN80JAa7bmPApOxOfaHy0SO",
    )

    runtimeconfig = WeaviateDocumentIndex.RuntimeConfig(batch_config=batch_config)
    store = WeaviateDocumentIndex[Document](db_config=dbconfig)
    store.configure(runtimeconfig)  # Batch settings being passed on
    store.index(docs)
