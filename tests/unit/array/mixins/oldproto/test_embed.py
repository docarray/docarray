import os

import numpy as np
import pytest
import tensorflow as tf

from docarray import DocumentArray
from docarray.array.annlite import DocumentArrayAnnlite
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic
from docarray.array.redis import DocumentArrayRedis


@pytest.fixture
def paddle_model():
    import paddle.nn as nn

    class DummyPaddleLayer(nn.Layer):
        def forward(self, x, y):
            return (x + y) / 2.0

    return DummyPaddleLayer()


def test_embeded_paddle_model(paddle_model):
    def collate_fn(da):
        return {'x': da.tensors, 'y': da.tensors}

    docs = DocumentArray.empty(3)
    docs.tensors = np.random.random([3, 5]).astype(np.float32)
    docs.embed(paddle_model, collate_fn=collate_fn, to_numpy=True)
    assert (docs.tensors == docs.embeddings).all()


random_embed_models = {
    'keras': lambda: tf.keras.Sequential(
        [tf.keras.layers.Dropout(0.5), tf.keras.layers.BatchNormalization()]
    ),
}

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    'framework, input_shape, embedding_shape',
    [
        ('keras', (128,), 128),
    ],
)
@pytest.mark.parametrize(
    'da_cls',
    [
        DocumentArray,
        DocumentArraySqlite,
        DocumentArrayAnnlite,
        DocumentArrayQdrant,
        # DocumentArrayWeaviate, TODO: enable this
        DocumentArrayElastic,
        DocumentArrayRedis,
    ],
)
@pytest.mark.parametrize('N', [2, 10])
@pytest.mark.parametrize('batch_size', [1, 256])
@pytest.mark.parametrize('to_numpy', [True, False])
def test_embedding_on_random_network(
    framework,
    input_shape,
    da_cls,
    embedding_shape,
    N,
    batch_size,
    to_numpy,
    start_storage,
):
    if da_cls in [
        DocumentArrayWeaviate,
        DocumentArrayAnnlite,
        DocumentArrayQdrant,
        DocumentArrayElastic,
        DocumentArrayRedis,
    ]:
        da = da_cls.empty(N, config={'n_dim': embedding_shape})
    else:
        da = da_cls.empty(N, config=None)
    da.tensors = np.random.random([N, *input_shape]).astype(np.float32)
    embed_model = random_embed_models[framework]()
    da.embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)

    r = da.embeddings
    if hasattr(r, 'numpy'):
        r = r.numpy()

    embed1 = r.copy()

    # reset
    da.embeddings = np.random.random([N, embedding_shape]).astype(np.float32)

    # docs[a: b].embed is only supported for DocumentArrayInMemory
    if isinstance(da, DocumentArrayInMemory):
        # try it again, it should yield the same result
        da.embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        np.testing.assert_array_almost_equal(da.embeddings, embed1)

        # reset
        da.embeddings = np.random.random([N, embedding_shape]).astype(np.float32)

        # now do this one by one
        da[: int(N / 2)].embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        da[-int(N / 2) :].embed(embed_model, batch_size=batch_size, to_numpy=to_numpy)
        np.testing.assert_array_almost_equal(da.embeddings, embed1)
