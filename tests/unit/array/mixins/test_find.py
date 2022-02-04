import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.math import ndarray
from docarray.array.storage.weaviate import WeaviateConfig


@pytest.mark.parametrize('storage, config', [('weaviate', WeaviateConfig(32))])
@pytest.mark.parametrize('limit', [1, 5, 10])
@pytest.mark.parametrize(
    'query',
    [np.random.random(32), np.random.random((1, 32)), np.random.random((2, 32))],
)
def test_find(storage, config, limit, query):
    embeddings = np.random.random((20, 32))

    if config:
        da = DocumentArray(storage=storage, config=config)
    else:
        da = DocumentArray(storage=storage)

    da.extend([Document(embedding=v) for v in embeddings])

    da_result = da.find(query, limit=limit)
    n_rows_query, _ = ndarray.get_array_rows(query)

    # check for each row on the query a DocumentArray is returned
    if n_rows_query == 1:
        assert len(da_result) == limit
    else:
        assert len(da_result) == n_rows_query

    # check returned objects are sorted
    if n_rows_query == 1:
        cosine_similarities = [t['cosine_similarity'] for t in da_result[:, 'tags']]
        assert sorted(cosine_similarities, reverse=True) == cosine_similarities
    else:
        for da in da_result:
            cosine_similarities = [t['cosine_similarity'] for t in da[:, 'tags']]
            assert sorted(cosine_similarities, reverse=True) == cosine_similarities
