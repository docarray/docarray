import pytest

from docarray import Document
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.storage.elastic import ElasticConfig


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=1)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=1)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=1)),
    ],
)
def test_insert(da_cls, config, start_storage):
    da = da_cls(config=config())
    assert not len(da)
    da.insert(0, Document(text='hello', id="0"))
    da.insert(0, Document(text='world', id="1"))
    assert len(da) == 2
    assert da[0].text == 'world'
    assert da[1].text == 'hello'
    assert da["1"].text == 'world'
    assert da["0"].text == 'hello'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=1)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=1)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=1)),
    ],
)
def test_append_extend(da_cls, config, start_storage):
    da = da_cls(config=config())
    da.append(Document())
    da.append(Document())
    assert len(da) == 2
    da.extend([Document(), Document()])
    assert len(da) == 4
