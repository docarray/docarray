import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.index]


class MyDoc(BaseDoc):
    tens: NdArray


def test_configure_dim(tmp_path):
    class Schema(BaseDoc):
        tens: NdArray = Field(dim=10)

    index = HnswDocumentIndex[Schema](work_dir=str(tmp_path))

    assert index._hnsw_indices['tens'].dim == 10

    docs = [Schema(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10


def test_configure_index(tmp_path):
    class Schema(BaseDoc):
        tens: NdArray[100] = Field(max_elements=12, space='cosine')
        tens_two: NdArray[10] = Field(M=4, space='ip')

    index = HnswDocumentIndex[Schema](work_dir=str(tmp_path))

    assert index._hnsw_indices['tens'].max_elements == 12
    assert index._hnsw_indices['tens'].space == 'cosine'
    assert index._hnsw_indices['tens'].M == 16  # default
    assert index._hnsw_indices['tens'].dim == 100

    assert index._hnsw_indices['tens_two'].max_elements == 1024  # default
    assert index._hnsw_indices['tens_two'].space == 'ip'
    assert index._hnsw_indices['tens_two'].M == 4
    assert index._hnsw_indices['tens_two'].dim == 10
