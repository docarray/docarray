import os

import numpy as np
import onnxruntime
import paddle
import pytest
import tensorflow as tf
import torch
from transformers import TFViTModel, ViTConfig, ViTModel

from docarray import DocumentArray
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.pqlite import DocumentArrayPqlite, PqliteConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate

random_embed_models = {
    'keras': lambda: tf.keras.Sequential(
        [tf.keras.layers.Dropout(0.5), tf.keras.layers.BatchNormalization()]
    ),
    'pytorch': lambda: torch.nn.Sequential(
        torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(128)
    ),
    'paddle': lambda: paddle.nn.Sequential(
        paddle.nn.Dropout(0.5), paddle.nn.BatchNorm1D(128)
    ),
    'transformers_torch': lambda: ViTModel(ViTConfig()),
    'transformers_tf': lambda: TFViTModel(ViTConfig()),
}
cur_dir = os.path.dirname(os.path.abspath(__file__))
torch.onnx.export(
    random_embed_models['pytorch'](),
    torch.rand(1, 128),
    os.path.join(cur_dir, 'test-net.onnx'),
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=['input'],  # the model's input names
    output_names=['output'],  # the model's output names
    dynamic_axes={
        'input': {0: 'batch_size'},  # variable length axes
        'output': {0: 'batch_size'},
    },
)

random_embed_models['onnx'] = lambda: onnxruntime.InferenceSession(
    os.path.join(cur_dir, 'test-net.onnx')
)


@pytest.mark.parametrize(
    'framework, input_shape, embedding_shape',
    [
        ('onnx', (128,), 128),
        ('keras', (128,), 128),
        ('pytorch', (128,), 128),
        ('transformers_torch', (3, 224, 224), 768),
        ('transformers_tf', (3, 224, 224), 768),
    ],
)
@pytest.mark.parametrize(
    'da_cls',
    [
        DocumentArray,
        DocumentArraySqlite,
        DocumentArrayPqlite,
        DocumentArrayQdrant,
        # DocumentArrayWeaviate, TODO: enable this
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
    if da_cls in [DocumentArrayWeaviate, DocumentArrayPqlite, DocumentArrayQdrant]:
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
