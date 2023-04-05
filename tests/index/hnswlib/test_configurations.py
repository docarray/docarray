import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDocument
from docarray.index import HnswDocumentIndex
from docarray.typing import NdArray

pytestmark = [pytest.mark.slow, pytest.mark.index]


class MyDoc(BaseDocument):
    tens: NdArray


def test_configure_dim(tmp_path):
    class Schema(BaseDocument):
        tens: NdArray = Field(dim=10)

    index = HnswDocumentIndex[Schema](work_dir=str(tmp_path))

    assert index._hnsw_indices['tens'].dim == 10

    docs = [Schema(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10
