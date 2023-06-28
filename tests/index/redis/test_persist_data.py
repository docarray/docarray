import numpy as np
import pytest
from pydantic import Field

from docarray import BaseDoc
from docarray.index import RedisDocumentIndex
from docarray.typing import NdArray
from tests.index.redis.fixtures import start_redis  # noqa: F401


pytestmark = [pytest.mark.slow, pytest.mark.index]


class SimpleDoc(BaseDoc):
    tens: NdArray[10] = Field(dim=1000)


def test_persist():
    query = SimpleDoc(tens=np.random.random((10,)))

    # create index
    index = RedisDocumentIndex[SimpleDoc](host='localhost')
    index_name = index._db_config.index_name

    assert index.num_docs() == 0

    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(10)])
    assert index.num_docs() == 10
    find_results_before = index.find(query, search_field='tens', limit=5)

    # load existing index
    index = RedisDocumentIndex[SimpleDoc](host='localhost', index_name=index_name)
    assert index.num_docs() == 10
    find_results_after = index.find(query, search_field='tens', limit=5)
    for doc_before, doc_after in zip(find_results_before[0], find_results_after[0]):
        assert doc_before.id == doc_after.id
        assert (doc_before.tens == doc_after.tens).all()

    # add new data
    index.index([SimpleDoc(tens=np.random.random((10,))) for _ in range(5)])
    assert index.num_docs() == 15
