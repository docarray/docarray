import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.redis import DocumentArrayRedis, RedisConfig
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig


@pytest.fixture(scope='function')
def docs():
    return [
        Document(text='hello'),
        Document(text='hello world'),
        Document(text='goodbye world!'),
    ]


@pytest.mark.parametrize('min_freq', [1, 2, 3])
@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, RedisConfig(n_dim=128)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=128)),
    ],
)
def test_da_vocabulary(da_cls, config, docs, min_freq, start_storage):
    if config:
        da = da_cls(docs, config=config)
    else:
        da = da_cls(docs)
    vocab = da.get_vocabulary(min_freq)
    if min_freq <= 1:
        assert set(vocab.values()) == {2, 3, 4}  # 0,1 are reserved
        assert set(vocab.keys()) == {'hello', 'world', 'goodbye'}
    elif min_freq == 2:
        assert set(vocab.values()) == {2, 3}  # 0,1 are reserved
        assert set(vocab.keys()) == {'hello', 'world'}
    elif min_freq == 3:
        assert not vocab.values()
        assert not vocab.keys()


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, RedisConfig(n_dim=128)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=128)),
    ],
)
def test_da_text_to_tensor_non_max_len(docs, da_cls, config, start_storage):
    if config:
        test_docs = da_cls(docs, config=config)
    else:
        test_docs = da_cls(docs)
    vocab = test_docs.get_vocabulary()
    test_docs.apply(lambda d: d.convert_text_to_tensor(vocab))
    np.testing.assert_array_equal(test_docs[0].tensor, [2])
    np.testing.assert_array_equal(test_docs[1].tensor, [2, 3])
    np.testing.assert_array_equal(test_docs[2].tensor, [4, 3])
    test_docs.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert test_docs[0].text == 'hello'
    assert test_docs[1].text == 'hello world'
    assert test_docs[2].text == 'goodbye world'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, RedisConfig(n_dim=128)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=128)),
    ],
)
def test_da_text_to_tensor_max_len_3(docs, da_cls, config, start_storage):
    if config:
        test_docs = da_cls(docs, config=config)
    else:
        test_docs = da_cls(docs)
    vocab = test_docs.get_vocabulary()
    test_docs.apply(lambda d: d.convert_text_to_tensor(vocab, max_length=3))

    np.testing.assert_array_equal(test_docs[0].tensor, [0, 0, 2])
    np.testing.assert_array_equal(test_docs[1].tensor, [0, 2, 3])
    np.testing.assert_array_equal(test_docs[2].tensor, [0, 4, 3])

    test_docs.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert test_docs[0].text == 'hello'
    assert test_docs[1].text == 'hello world'
    assert test_docs[2].text == 'goodbye world'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, RedisConfig(n_dim=128)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=128)),
    ],
)
def test_da_text_to_tensor_max_len_1(docs, da_cls, config, start_storage):
    if config:
        test_docs = da_cls(docs, config=config)
    else:
        test_docs = da_cls(docs)
    vocab = test_docs.get_vocabulary()
    test_docs.apply(lambda d: d.convert_text_to_tensor(vocab, max_length=1))

    np.testing.assert_array_equal(test_docs[0].tensor, [2])
    np.testing.assert_array_equal(test_docs[1].tensor, [3])
    np.testing.assert_array_equal(test_docs[2].tensor, [3])

    test_docs.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert test_docs[0].text == 'hello'
    assert test_docs[1].text == 'world'
    assert test_docs[2].text == 'world'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, RedisConfig(n_dim=128)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=128)),
    ],
)
def test_convert_text_tensor_random_text(da_cls, docs, config, start_storage):
    if config:
        da = da_cls(docs, config=config)
    else:
        da = da_cls(docs)
    texts = ['a short phrase', 'word', 'this is a much longer sentence']
    da.clear()
    da.extend(Document(text=t) for t in texts)
    vocab = da.get_vocabulary()

    # encoding
    da.apply(lambda d: d.convert_text_to_tensor(vocab, max_length=10))

    # decoding
    da.apply(lambda d: d.convert_tensor_to_text(vocab))

    assert texts
    assert da.texts == texts
