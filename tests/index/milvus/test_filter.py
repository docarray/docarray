import numpy as np
from pydantic import Field

from docarray import BaseDoc
from docarray.index import MilvusDocumentIndex
from docarray.typing import NdArray


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]
    number: int


def test_filter_range():  # noqa: F811
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        number: int

    index = MilvusDocumentIndex[SimpleSchema](index_name="embedding")

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            number=i,
        )
        for i in range(10)
    ]
    index.index(index_docs)

    docs = index.filter("number > 8", limit=5)

    assert len(docs) == 1

    docs = index.filter(f"id == '{index_docs[0].id}'", limit=5)
    assert docs[0].id == index_docs[0].id
