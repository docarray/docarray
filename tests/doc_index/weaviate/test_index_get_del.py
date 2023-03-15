import numpy as np
import pytest
import weaviate
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.weaviate_doc_index import WeaviateDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000, is_embedding=True)


@pytest.fixture
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


@pytest.fixture
def weaviate_client():
    client = weaviate.Client("http://weaviate:8080")
    client.schema.delete_all()
    yield client
    client.schema.delete_all()


def test_index_simple_schema(weaviate_client, ten_simple_docs):
    store = WeaviateDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)
    assert store.num_docs() == 10

    for doc in ten_simple_docs:
        doc_id = doc.id
        doc_embedding = doc.tens

        result = (
            weaviate_client.query.get("Document", "__id")
            .with_additional("vector")
            .with_where({"path": ["__id"], "operator": "Equal", "valueString": doc_id})
            .do()
        )

        result = result["data"]["Get"]["Document"][0]
        assert result["__id"] == doc_id
        assert np.allclose(result["_additional"]["vector"], doc_embedding)
