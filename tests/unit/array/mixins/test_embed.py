import os

import numpy as np
import onnxruntime
import pytest
import torch
from transformers import (
    TFViTModel,
    ViTConfig,
    ViTModel,
    BertModel,
    BertConfig,
    TFBertModel,
)

from docarray import DocumentArray
from docarray.array.annlite import DocumentArrayAnnlite
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic
from docarray.array.redis import DocumentArrayRedis

random_embed_models = {
    'pytorch': lambda: torch.nn.Sequential(
        torch.nn.Dropout(0.5), torch.nn.BatchNorm1d(128)
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


@pytest.fixture
def bert_tokenizer(tmpfile):
    from transformers import BertTokenizer

    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    with open(tmpfile, "w", encoding="utf-8") as vocab_writer:
        vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
    return BertTokenizer(tmpfile)


@pytest.mark.parametrize(
    'bert_transformer, return_tensors',
    [(BertModel(BertConfig()), 'pt'), (TFBertModel(BertConfig()), 'tf')],
)
def test_embed_bert_model(bert_transformer, bert_tokenizer, return_tensors):
    def collate_fn(da):
        return bert_tokenizer(
            da.texts,
            return_tensors=return_tensors,
        )

    docs = DocumentArray.empty(1)
    docs[0].text = 'this is some random text to embed'
    docs.embed(bert_transformer, collate_fn=collate_fn)
    assert list(docs.embeddings.shape) == [1, 768]
