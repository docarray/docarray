import pytest

from docarray import DocumentArray
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.pqlite import DocumentArrayPqlite, PqliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayPqlite, PqliteConfig(n_dim=5)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=5)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=5)),
    ],
)
def test_empty_non_zero(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(10, config=config)
    else:
        da = da_cls.empty(10)
    da = DocumentArray.empty(10)
    assert len(da) == 10
    da = DocumentArray.empty()
    assert len(da) == 0
