from pydantic import Field

from docarray import BaseDoc
from docarray.index import QdrantDocumentIndex
from docarray.typing import NdArray

from .fixtures import qdrant_config, qdrant


def test_dict_filter_get_passed(qdrant_config, qdrant):
    class SimpleSchema(BaseDoc):
        embedding: NdArray[10] = Field(space='cosine')  # type: ignore[valid-type]
        text: str

    store = QdrantDocumentIndex[SimpleSchema](db_config=qdrant_config)

    query = {
        'filter': {'must': [{'key': 'city', 'match': {'value': 'London'}}]},
        'params': {'hnsw_ef': 128, 'exact': False},
        'vector': [0.2, 0.1, 0.9, 0.7],
        'limit': 3,
    }

    points = store.execute_query(query=query)
    assert points is not None
