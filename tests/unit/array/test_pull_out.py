import numpy as np
import pytest

from docarray import DocumentArray, Document


@pytest.fixture(scope='function')
def docs():
    d1 = Document(embedding=np.array([10, 0]))
    d2 = Document(embedding=np.array([0, 10]))
    d3 = Document(embedding=np.array([-10, -10]))
    yield d1, d2, d3


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', {'n_dim': 2}),
        ('pqlite', {'n_dim': 2}),
        ('qdrant', {'n_dim': 2}),
    ],
)
def test_update_embedding(docs, storage, config, start_storage):
    if config:
        da = DocumentArray(docs, storage=storage, config=config)
    else:
        da = DocumentArray(docs, storage=storage)

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[1].id
    assert results[1].id == docs[0].id
    assert results[2].id == docs[2].id

    da[0, 'embedding'] = np.array([1.1, 9.1])

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[0].id
    assert results[1].id == docs[1].id
    assert results[2].id == docs[2].id

    assert np.testing.assert_almost_equal(da[0].embedding, np.array([1.1, 9.1]))


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', {'n_dim': 2}),
        ('pqlite', {'n_dim': 2}),
        ('qdrant', {'n_dim': 2}),
    ],
)
def test_update_doc_embedding(docs, storage, config, start_storage):
    if config:
        da = DocumentArray(docs, storage=storage, config=config)
    else:
        da = DocumentArray(docs, storage=storage)

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[1].id
    assert results[1].id == docs[0].id
    assert results[2].id == docs[2].id

    da[0] = Document(id=docs[0].id, embedding=np.array([1.1, 9.1]))

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[0].id
    assert results[1].id == docs[1].id
    assert results[2].id == docs[2].id

    assert np.testing.assert_almost_equal(da[0].embedding, np.array([1.1, 9.1]))


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', {'n_dim': 2}),
        ('pqlite', {'n_dim': 2}),
        ('qdrant', {'n_dim': 2}),
    ],
)
def test_batch_update_embedding(docs, storage, config, start_storage):
    if config:
        da = DocumentArray(docs, storage=storage, config=config)
    else:
        da = DocumentArray(docs, storage=storage)

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[1].id
    assert results[1].id == docs[0].id
    assert results[2].id == docs[2].id

    da[:, 'embedding'] = np.array([[0, 10], [10, 0], [-10, -10]])

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[0].id
    assert results[1].id == docs[1].id
    assert results[2].id == docs[2].id

    assert np.testing.assert_almost_equal(da[0].embedding, np.array([0, 10]))


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', {'n_dim': 2}),
        ('pqlite', {'n_dim': 2}),
        ('qdrant', {'n_dim': 2}),
    ],
)
def test_batch_update_doc_embedding(docs, storage, config, start_storage):
    if config:
        da = DocumentArray(docs, storage=storage, config=config)
    else:
        da = DocumentArray(docs, storage=storage)

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[1].id
    assert results[1].id == docs[0].id
    assert results[2].id == docs[2].id

    da[:2] = [
        Document(id=docs[0].id, embedding=np.array([0, 10])),
        Document(id=docs[1].id, embedding=np.array([10, 0])),
    ]

    results = da.find(np.array([1, 9]))
    assert results[0].id == docs[0].id
    assert results[1].id == docs[1].id
    assert results[2].id == docs[2].id

    assert np.testing.assert_almost_equal(da[0].embedding, np.array([0, 10]))
