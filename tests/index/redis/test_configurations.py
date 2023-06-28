import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray
from tests.index.redis.fixtures import start_redis  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


def test_configure_dim():
    class Schema(BaseDoc):
        tens: NdArray = Field(dim=10)

    index = RedisDocumentIndex[Schema](host='localhost')

    docs = [Schema(tens=np.random.random((10,))) for _ in range(10)]
    index.index(docs)

    assert index.num_docs() == 10


def test_configure_index(tmp_path):
    class Schema(BaseDoc):
        tens: NdArray[100] = Field(space='cosine')
        title: str
        year: int

    types = {'id': 'TEXT', 'tens': 'VECTOR', 'title': 'TEXT', 'year': 'NUMERIC'}
    index = RedisDocumentIndex[Schema](host='localhost')

    attr_bytes = index._client.ft(index._index_name).info()['attributes']
    attr = [[byte.decode() for byte in sublist] for sublist in attr_bytes]

    assert len(Schema.__fields__) == len(attr)
    for field, attr in zip(Schema.__fields__, attr):
        assert field in attr and types[field] in attr
