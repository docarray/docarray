import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray
from tests.index.milvus.fixtures import start_storage  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_configure_dim():
    class Schema1(BaseDoc):
        tens: NdArray[10] = Field(is_embedding=True)

    index = MilvusDocumentIndex[Schema1]()

    docs = [Schema1(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10

    class Schema2(BaseDoc):
        tens: NdArray = Field(is_embedding=True, dim=10)

    index = MilvusDocumentIndex[Schema2]()

    docs = [Schema2(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10

    class Schema3(BaseDoc):
        tens: NdArray = Field(is_embedding=True)

    with pytest.raises(ValueError, match='The dimension information is missing'):
        MilvusDocumentIndex[Schema3]()


def test_incorrect_vector_field():
    class Schema1(BaseDoc):
        tens: NdArray[10]

    with pytest.raises(ValueError, match='Unable to find any vector columns'):
        MilvusDocumentIndex[Schema1]()

    class Schema2(BaseDoc):
        tens1: NdArray[10] = Field(is_embedding=True)
        tens2: NdArray[20] = Field(is_embedding=True)

    with pytest.raises(
        ValueError, match='Specifying multiple vector fields is not supported'
    ):
        MilvusDocumentIndex[Schema2]()
