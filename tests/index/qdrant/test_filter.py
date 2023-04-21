import numpy as np
from pydantic import Field
from qdrant_client.http import models as rest

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import qdrant, qdrant_config  # noqa: F401


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]
    number: int


def test_filter_range(qdrant_config):  # noqa: F811
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        number: int

    index = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            number=i,
        )
        for i in range(10)
    ]
    index.index(index_docs)

    filter_query = rest.Filter(
        must=[rest.FieldCondition(key='number', range=rest.Range(gte=5, lte=7))]
    )
    docs = index.filter(filter_query, limit=5)

    assert len(docs) == 3
