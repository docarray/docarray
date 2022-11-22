import pytest

from docarray import DocumentArray
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.redis import DocumentArrayRedis, RedisConfig
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig


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
def test_sample(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(100, config=config)
    else:
        da = da_cls.empty(100)
    sampled = da.sample(1)
    assert len(sampled) == 1
    sampled = da.sample(5)
    assert len(sampled) == 5
    assert isinstance(sampled, DocumentArray)
    with pytest.raises(ValueError):
        da.sample(101)  # can not sample with k greater than lenth of document array.


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
def test_sample_with_seed(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(100, config=config)
    else:
        da = da_cls.empty(100)
    sampled_1 = da.sample(5, seed=1)
    sampled_2 = da.sample(5, seed=1)
    sampled_3 = da.sample(5, seed=2)
    assert len(sampled_1) == len(sampled_2) == len(sampled_3) == 5
    assert sampled_1 == sampled_2
    assert sampled_1 != sampled_3


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
def test_shuffle(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(100, config=config)
    else:
        da = da_cls.empty(100)
    shuffled = da.shuffle()
    assert len(shuffled) == len(da)
    assert isinstance(shuffled, DocumentArray)
    ids_before_shuffle = [d.id for d in da]
    ids_after_shuffle = [d.id for d in shuffled]
    assert ids_before_shuffle != ids_after_shuffle
    assert sorted(ids_before_shuffle) == sorted(ids_after_shuffle)


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
def test_shuffle_with_seed(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(100, config=config)
    else:
        da = da_cls.empty(100)
    shuffled_1 = da.shuffle(seed=1)
    shuffled_2 = da.shuffle(seed=1)
    shuffled_3 = da.shuffle(seed=2)
    assert len(shuffled_1) == len(shuffled_2) == len(shuffled_3) == len(da)
    assert shuffled_1 == shuffled_2
    assert shuffled_1 != shuffled_3
