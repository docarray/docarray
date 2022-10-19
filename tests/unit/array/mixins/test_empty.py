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


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=5)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=5)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=5)),
        (DocumentArrayElastic, ElasticConfig(n_dim=5)),
        (DocumentArrayRedis, RedisConfig(n_dim=5)),
    ],
)
def test_empty_non_zero(da_cls, config, start_storage):
    # Assert .empty provides a da with 0 docs
    if config:
        da = da_cls.empty(config=config)
    else:
        da = da_cls.empty()

    assert len(da) == 0
    if da_cls == DocumentArrayAnnlite:
        da._annlite.close()

    # Assert .empty provides a da of the correct length
    if config:
        da = da_cls.empty(10, config=config)
    else:
        da = da_cls.empty(10)

    assert len(da) == 10
