import numpy as np
import pytest
import tensorflow as tf

from docarray import DocumentArray, Document

rand_array = np.random.random([10, 3])


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', {}),
        ('sqlite', {}),
        ('annlite', {'n_dim': 3}),
        ('weaviate', {'n_dim': 3}),
        ('qdrant', {'n_dim': 3}),
        ('elasticsearch', {'n_dim': 3}),
        ('redis', {'n_dim': 3}),
    ],
)
@pytest.mark.parametrize(
    'array',
    [
        tf.constant(rand_array),
    ],
)
def test_set_embeddings_multi_kind(array, storage, config, start_storage):
    da = DocumentArray([Document() for _ in range(10)], storage=storage, config=config)
    da[:, 'embedding'] = array
