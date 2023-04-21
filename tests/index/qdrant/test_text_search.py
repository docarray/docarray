import numpy as np
from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import qdrant, qdrant_config  # noqa: F401


class SimpleDoc(BaseDoc):
    embedding: NdArray[10] = Field(dim=1000)  # type: ignore[valid-type]
    text: str


def test_text_search(qdrant_config):  # noqa: F811
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        text: str

    index = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            text=f'Lorem ipsum {i}',
        )
        for i in range(10)
    ]
    index.index(index_docs)

    query = 'ipsum 2'
    docs, scores = index.text_search(query, search_field='text', limit=5)

    assert len(docs) == 1
    assert len(scores) == 1
    assert docs[0].id == index_docs[2].id
    assert scores[0] > 0.0


def test_text_search_batched(qdrant_config):  # noqa: F811
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        text: str

    index = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    index_docs = [
        SimpleDoc(
            embedding=np.zeros(10),
            text=f'Lorem ipsum {i}',
        )
        for i in range(10)
    ]
    index.index(index_docs)

    queries = ['ipsum 2', 'ipsum 4', 'Lorem']
    docs, scores = index.text_search_batched(queries, search_field='text', limit=5)

    assert len(docs) == 3
    assert len(docs[0]) == 1
    assert len(docs[1]) == 1
    assert len(docs[2]) == 5
    assert len(scores) == 3
    assert len(scores[0]) == 1
    assert len(scores[1]) == 1
    assert len(scores[2]) == 5
