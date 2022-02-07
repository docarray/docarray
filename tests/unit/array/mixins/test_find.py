import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.math import ndarray


@pytest.mark.parametrize(
    'storage, config', [('weaviate', WeaviateConfig(32)), ('pqlite', {'n_dim': 32})]
)
@pytest.mark.parametrize('limit', [1, 5, 10])
@pytest.mark.parametrize(
    'query',
    [np.random.random(32), np.random.random((1, 32)), np.random.random((2, 32))],
)
def test_find(storage, config, limit, query, start_weaviate):
    embeddings = np.random.random((20, 32))

    if config:
        da = DocumentArray(storage=storage, config=config)
    else:
        da = DocumentArray(storage=storage)

    da.extend([Document(embedding=v) for v in embeddings])

    da_result = da.find(query, limit=limit)
    n_rows_query, _ = ndarray.get_array_rows(query)

    print(f'\n\n\ntype(da_result)={type(da_result)}')

    # check for each row on the query a DocumentArray is returned
    if n_rows_query == 1:
        assert len(da_result) == limit
    else:
        assert len(da_result) == n_rows_query

    # check returned objects are sorted according to the storage backend metric
    # weaviate uses cosine similarity by default
    # pqlite uses cosine distance by default
    if n_rows_query == 1:
        if storage == 'weaviate':
            cosine_similarities = [
                t['cosine_similarity'].value for t in da_result[:, 'scores']
            ]
            assert sorted(cosine_similarities, reverse=True) == cosine_similarities
        elif storage == 'weaviate':
            cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
            assert sorted(cosine_distances, reverse=False) == cosine_distances
    else:
        if storage == 'weaviate':
            for da in da_result:
                cosine_similarities = [
                    t['cosine_similarity'].value for t in da[:, 'scores']
                ]
                assert sorted(cosine_similarities, reverse=True) == cosine_similarities
        elif storage == 'pqlite':
            for da in da_result:
                cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
                assert sorted(cosine_distances, reverse=False) == cosine_distances
