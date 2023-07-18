import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import start_storage, tmp_collection_name  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_configure_dim():
    class Schema1(BaseDoc):
        tens: NdArray = Field(dim=10)

    index = QdrantDocumentIndex[Schema1](host='localhost')

    docs = [Schema1(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10

    class Schema2(BaseDoc):
        tens: NdArray[20]

    index = QdrantDocumentIndex[Schema2](host='localhost')
    docs = [Schema2(tens=np.random.random((20,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10


def test_index_name():
    class Schema(BaseDoc):
        tens: NdArray = Field(dim=10)

    index1 = QdrantDocumentIndex[Schema]()
    assert index1.index_name == 'schema'

    index2 = QdrantDocumentIndex[Schema](index_name='my_index')
    assert index2.index_name == 'my_index'

    index3 = QdrantDocumentIndex[Schema](collection_name='my_index')
    assert index3.index_name == 'my_index'
