import numpy as np
import pytest

from docarray import BaseDoc
from docarray.index import RedisDocumentIndex
from pydantic import Field
from docarray.typing import NdArray
from tests.index.redis.fixtures import start_redis, tmp_collection_name  # noqa: F401
from typing import Optional


@pytest.mark.parametrize('space', ['cosine'])
def test_find_simple_schema(space):
    class SimpleSchema(BaseDoc):
        tens: Optional[NdArray[10]] = Field(space=space, algorithm='HNSW')  # type: ignore[valid-type]
        bla: int
        title: str
        smth: Optional[str] = None
        tenss: Optional[NdArray[10]] = None

    index = RedisDocumentIndex[SimpleSchema](host='localhost')

    docs = [SimpleSchema(bla=i, title=f'zdall {i}', tens=np.random.rand(10)) for i in range(5)]
    docs.append(SimpleSchema(bla=6, title=f'hey everyone how are you', tens=np.random.rand(10)))
    docs.append(SimpleSchema(bla=7, title=f'hey how are you', tens=np.random.rand(10)))


    index.index(docs)

    query = np.random.rand(10)
    results = index.find(query, search_field='tens')
    print(len(results))

    results = index.find_batched(np.array([np.random.rand(10), np.random.rand(10)]), search_field='tens')
    print('find batched', results)
    res = index[docs[0].id]
    print(index.num_docs())
    del index[docs[0].id]
    print(index.num_docs())

    docs = index.filter({'bla': {'$gt': 3}})

    print('filtered', docs)

    docs = index.filter_batched([{'bla': {'$gt': 3}}, {'bla': {'$lte': 3}}])
    print('batched filt', docs)

    docs = index.text_search(query='hey everyone', search_field='title')
    print(docs)

    docs = index.text_search_batched(queries=['hey hey', 'hey everyone'], search_field='title')
    print(docs)


def test_simple_scenario():
    # Define a document schema
    class SimpleSchema(BaseDoc):
        tensor: Optional[NdArray[10]] = Field(space='COSINE')
        year: int
        title: Optional[str] = None

    # Create a document index
    index = RedisDocumentIndex[SimpleSchema](host='localhost')

    # Prepare documents
    docs = [SimpleSchema(year=i, title=f'some text {i}', tensor=np.random.rand(10)) for i in range(5)]

    # Index
    index.index(docs)

    # Search
    query = np.random.rand(10)
    results = index.find(query, search_field='tensor')
    print(results)
