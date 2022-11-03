import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.annlite import AnnliteConfig
from docarray.array.qdrant import QdrantConfig
from docarray.array.elastic import ElasticConfig
from docarray.array.redis import RedisConfig


@pytest.fixture
def docs():
    yield (Document(text=str(j)) for j in range(100))


@pytest.fixture
def indices():
    yield (i for i in [-2, 0, 2])


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('weaviate', WeaviateConfig(n_dim=123)),
        ('annlite', AnnliteConfig(n_dim=123, enable_offset2id=False)),
        ('qdrant', QdrantConfig(n_dim=123)),
        ('elasticsearch', ElasticConfig(n_dim=123)),
        ('redis', RedisConfig(n_dim=123)),
    ],
)
def test_disable_offset2id_flag(docs, storage, config, start_storage):
    if config:
        docs = DocumentArray(docs, storage=storage, config=config)
    else:
        docs = DocumentArray(docs, storage=storage)
    # getter
    assert docs[99].text == '99'
    assert docs[np.int(99)].text == '99'
    assert docs[-1].text == '99'
    assert docs[0].text == '0'
    # string index
    assert docs[docs[0].id].text == '0'
    assert docs[docs[99].id].text == '99'
    assert docs[docs[-1].id].text == '99'

    with pytest.raises(IndexError):
        docs[100]

    with pytest.raises(KeyError):
        docs['adsad']
