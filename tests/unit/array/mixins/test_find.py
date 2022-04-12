import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.math import ndarray


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 32}),
        ('annlite', {'n_dim': 32}),
        ('qdrant', {'n_dim': 32}),
        ('elasticsearch', {'n_dim': 32}),
    ],
)
@pytest.mark.parametrize('limit', [1, 5, 10])
@pytest.mark.parametrize(
    'query',
    [np.random.random(32), np.random.random((1, 32)), np.random.random((2, 32))],
)
def test_find(storage, config, limit, query, start_storage):
    embeddings = np.random.random((20, 32))

    if config:
        da = DocumentArray(storage=storage, config=config)
    else:
        da = DocumentArray(storage=storage)

    da.extend([Document(embedding=v) for v in embeddings])

    result = da.find(query, limit=limit)
    n_rows_query, _ = ndarray.get_array_rows(query)

    # check for each row on the query a DocumentArray is returned
    if n_rows_query == 1:
        assert len(result) == limit
    else:
        assert len(result) == n_rows_query

    # check returned objects are sorted according to the storage backend metric
    # weaviate uses cosine similarity by default
    # annlite uses cosine distance by default
    if n_rows_query == 1:
        if storage == 'weaviate':
            cosine_similarities = [
                t['cosine_similarity'].value for t in result[:, 'scores']
            ]
            assert sorted(cosine_similarities, reverse=True) == cosine_similarities
        elif storage in ['memory', 'annlite', 'elasticsearch']:
            cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
            assert sorted(cosine_distances, reverse=False) == cosine_distances
    else:
        if storage == 'weaviate':
            for da in result:
                cosine_similarities = [
                    t['cosine_similarity'].value for t in da[:, 'scores']
                ]
                assert sorted(cosine_similarities, reverse=True) == cosine_similarities
        elif storage in ['memory', 'annlite', 'elasticsearch']:
            for da in result:
                cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
                assert sorted(cosine_distances, reverse=False) == cosine_distances


@pytest.mark.parametrize(
    'storage, config',
    [
        ('elasticsearch', {'n_dim': 32, 'index_text': True}),
    ],
)
def test_find_by_text(storage, config, start_storage):
    da = DocumentArray(storage=storage, config=config)
    da.extend(
        [
            Document(id='1', text='token1 token2 token3'),
            Document(id='2', text='token1 token2'),
            Document(id='3', text='token2 token3 token4'),
        ]
    )

    results = da.find('token1')
    assert isinstance(results, DocumentArray)
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '2'}

    results = da.find(['token1'])
    # assert isinstance(results, list) # not until find is changed across .find
    assert len(results) == 1
    assert set(results[0][:, 'id']) == {'1', '2'}

    results = da.find('token2 token3')
    assert isinstance(results, DocumentArray)
    assert len(results) == 3
    assert set(results[:, 'id']) == {'1', '2', '3'}

    results = da.find('token3 token4')
    assert isinstance(results, DocumentArray)
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '3'}

    results = da.find(['token4', 'token'])
    assert isinstance(results, list)
    assert len(results) == 2  # len(input) = len(output)
    assert len(results[0]) == 1  # 'token4' only appears in one doc
    assert results[0][0].id == '3'  # 'token4' only appears in doc3
    assert len(results[1]) == 0  # 'token' is not present in da vocabulary


@pytest.mark.parametrize(
    'storage, config',
    [
        ('elasticsearch', {'n_dim': 32, 'tag_indices': ['attr1', 'attr2', 'attr3']}),
    ],
)
def test_find_by_tag(storage, config):
    da = DocumentArray(storage=storage, config=config)
    da.extend(
        [
            Document(
                id='1',
                tags={
                    'attr1': 'token1 token2 token3',
                    'attr2': 'token2 token3 token4',
                    'attr3': 'token4 token5 token6',
                },
            ),
            Document(
                id='2',
                tags={
                    'attr1': 'token1',
                    'attr2': 'token2',
                    'attr3': 'token6',
                },
            ),
            Document(
                id='3',
                tags={
                    'attr1': 'token4',
                    'attr2': 'token3',
                    'attr3': 'token1 token5',
                },
            ),
            Document(id='4'),
        ]
    )

    results = da.find('token1 token2', index='attr1')
    assert len(results) == 2
    assert results[0].id == '1'
    assert results[1].id == '2'

    results = da.find('token2 token4', index='attr1')
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '3'}

    results = da.find('token4', index='attr2')
    assert len(results) == 1
    assert results[0].id == '1'

    results = da.find('token6', index='attr3')
    assert len(results) == 2
    assert results[0].id == '2'
    assert results[1].id == '1'

    results = da.find('token5', index='attr3')
    assert len(results) == 2
    assert set(results[:, 'id']) == {'1', '3'}
    assert all(['token5' in r.tags['attr3'] for r in results]) == True

    results = da.find('token1', index='attr3')
    assert len(results) == 1
    assert results[0].id == '3'
    assert all(['token1' in r.tags['attr3'] for r in results]) == True
