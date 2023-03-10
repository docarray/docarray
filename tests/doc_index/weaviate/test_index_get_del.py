import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument
from docarray.doc_index.backends.weaviate_doc_index import WeaviateDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDocument):
    tens: NdArray[10] = Field(dim=1000)


@pytest.fixture
def ten_simple_docs():
    return [SimpleDoc(tens=np.random.randn(10)) for _ in range(10)]


def test_index_simple_schema(ten_simple_docs):
    store = WeaviateDocumentIndex[SimpleDoc]()
    store.index(ten_simple_docs)
    assert store.num_docs() == 10
