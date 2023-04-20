from typing import Optional, Sequence

import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray
from tests.index.qdrant.fixtures import qdrant, qdrant_config  # noqa: F401


class SimpleDoc(BaseDoc):
    embedding: NdArray[4] = Field(space='cosine')  # type: ignore[valid-type]
    text: Optional[str]


@pytest.fixture
def index_docs() -> Sequence[SimpleDoc]:
    index_docs = [SimpleDoc(embedding=np.zeros(4), text=f'Test {i}') for i in range(10)]
    return index_docs


@pytest.mark.parametrize('limit', [1, 5, 10])
def test_dict_limit(qdrant_config, index_docs, limit):  # noqa: F811
    index = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    index.index(index_docs)

    # Search test
    query = {
        'vector': ('embedding', [1.0, 0.0, 0.0, 0.0]),
        'limit': limit,
        'with_vectors': True,
    }

    points = index.execute_query(query=query)
    assert points is not None
    assert len(points) == limit

    # Scroll test
    query = {
        'limit': limit,
        'with_vectors': True,
    }

    points = index.execute_query(query=query)
    assert points is not None
    assert len(points) == limit


def test_dict_full_text_filter(qdrant_config, index_docs):  # noqa: F811
    index = QdrantDocumentIndex[SimpleDoc](db_config=qdrant_config)
    index.index(index_docs)

    # Search test
    query = {
        'filter': {'must': [{'key': 'text', 'match': {'text': '2'}}]},
        'params': {'hnsw_ef': 128, 'exact': False},
        'vector': ('embedding', [1.0, 0.0, 0.0, 0.0]),
        'limit': 3,
        'with_vectors': True,
    }

    points = index.execute_query(query=query)
    assert points is not None
    assert len(points) == 1
    assert points[0].id == index_docs[2].id

    # Scroll test
    query = {
        'filter': {'must': [{'key': 'text', 'match': {'text': '2'}}]},
        'params': {'hnsw_ef': 128, 'exact': False},
        'limit': 3,
        'with_vectors': True,
    }

    points = index.execute_query(query=query)
    assert points is not None
    assert len(points) == 1
    assert points[0].id == index_docs[2].id
