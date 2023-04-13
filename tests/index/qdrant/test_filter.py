import pytest
import qdrant_client
import numpy as np

from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray

from qdrant_client.http import models as rest

from .fixtures import qdrant_config, qdrant  # ignore: type[import]


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]
    number: int


def test_filter_range(qdrant_config, qdrant):
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        number: int

    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            number=i,
        )
        for i in range(10)
    ]
    store.index(index_docs)

    filter_query = rest.Filter(
        must=[rest.FieldCondition(key='number', range=rest.Range(gte=5, lte=7))]
    )
    docs = store.filter(filter_query, limit=5)

    assert len(docs) == 3
