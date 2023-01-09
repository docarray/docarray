import tempfile
import uuid

import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.array.elastic import DocumentArrayElastic
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.opensearch import DocumentArrayOpenSearch
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.redis import DocumentArrayRedis
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.storage.elastic import ElasticConfig
from docarray.array.storage.opensearch import OpenSearchConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.redis import RedisConfig
from docarray.array.storage.sqlite import SqliteConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig
from tests.conftest import tmpfile


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=1)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=1)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=1)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=1)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=128)),
        (DocumentArrayOpenSearch, lambda: OpenSearchConfig(n_dim=1)),
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
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=1)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=128)),
    ],
)
def test_append_extend(da_cls, config, start_storage):
    da = da_cls(config=config())
    da.append(Document())
    da.append(Document())
    assert len(da) == 2
    # assert len(da._offset2ids.ids) == 2 will not work unless used in a context manager
    da.extend([Document(), Document()])
    assert len(da) == 4
    # assert len(da._offset2ids.ids) == 4 will not work unless used in a context manager


def update_config_inplace(config, tmpdir, tmpfile):
    variable_names = ['table_name', 'connection', 'collection_name', 'index_name']
    variable_names_db = ['connection']

    for field in variable_names_db:
        if field in config:
            config[field] = str(tmpfile)

    for field in variable_names:
        if field in config:
            config[field] = f'{config[field]}_{uuid.uuid4().hex}'


@pytest.mark.parametrize(
    'storage, config',
    [
        ('sqlite', {'table_name': 'Test', 'connection': 'sqlite'}),
        ('weaviate', {'n_dim': 3, 'name': 'Weaviate'}),
        ('qdrant', {'n_dim': 3, 'collection_name': 'qdrant'}),
        ('elasticsearch', {'n_dim': 3, 'index_name': 'elasticsearch'}),
        ('opensearch', {'n_dim': 3, 'index_name': 'opensearch'}),
        ('redis', {'n_dim': 3, 'index_name': 'redis'}),
        ('milvus', {'n_dim': 3, 'collection_name': 'redis'}),
    ],
)
def test_context_manager_from_disk(storage, config, start_storage, tmpdir, tmpfile):
    config = config
    update_config_inplace(config, tmpdir, tmpfile)

    da = DocumentArray(storage=storage, config=config)

    with da as da_open:
        da_open.append(Document(embedding=np.random.random(3)))
        da_open.append(Document(embedding=np.random.random(3)))

    assert len(da) == 2
    assert len(da._offset2ids.ids) == 2

    da2 = DocumentArray(storage=storage, config=config)

    assert len(da2) == 2
    assert len(da2._offset2ids.ids) == 2

    # Cleanup modifications made in test
    with da:
        del da[0]
        del da[0]


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2'}),
        ('milvus', {'n_dim': 3, 'distance': 'L2'}),
        ('opensearch', {'n_dim': 3, 'distance': 'l2'}),
    ],
)
def test_extend_subindex(storage, config, start_storage):

    n_dim = 3
    subindex_configs = (
        {'@c': dict()} if storage in ['sqlite', 'memory'] else {'@c': {'n_dim': 2}}
    )
    da = DocumentArray(
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    with da:
        da.extend(
            [
                Document(
                    id=str(i),
                    embedding=i * np.ones(n_dim),
                    chunks=[
                        Document(id=str(i) + '_0', embedding=np.array([i, i])),
                        Document(id=str(i) + '_1', embedding=np.array([i, i])),
                    ],
                )
                for i in range(3)
            ]
        )

    assert len(da._subindices['@c']) == 6

    for j in range(2):
        for i in range(3):
            assert (da._subindices['@c'][f'{i}_{j}'].embedding == [i, i]).all()


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2'}),
        ('milvus', {'n_dim': 3, 'distance': 'L2'}),
        ('opensearch', {'n_dim': 3, 'distance': 'l2'}),
    ],
)
def test_append_subindex(storage, config, start_storage):

    n_dim = 3
    subindex_configs = (
        {'@c': dict()} if storage in ['sqlite', 'memory'] else {'@c': {'n_dim': 2}}
    )
    da = DocumentArray(
        storage=storage,
        config=config,
        subindex_configs=subindex_configs,
    )

    with da:
        da.append(
            Document(
                embedding=np.ones(n_dim),
                chunks=[
                    Document(id='0', embedding=np.array([0, 0])),
                    Document(id='1', embedding=np.array([1, 1])),
                ],
            )
        )

    with da:
        assert len(da._subindices['@c']) == 2

        for i in range(2):
            assert embeddings_eq(da._subindices['@c'][f'{i}'].embedding, [i, i])


def embeddings_eq(emb1, emb2):
    b = emb1 == emb2
    if isinstance(b, bool):
        return b
    else:
        return b.all()


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2'}),
        ('milvus', {'n_dim': 3, 'distance': 'L2'}),
        ('opensearch', {'n_dim': 3, 'distance': 'l2'}),
    ],
)
@pytest.mark.parametrize(
    'index', [1, '1', slice(1, 2), [1], [False, True, False, False, False]]
)
def test_del_and_append(index, storage, config, start_storage):
    da = DocumentArray(storage=storage, config=config)

    with da:
        da.extend([Document(id=str(i)) for i in range(5)])
    with da:
        del da[index]
        da.append(Document(id='new'))

    assert da[:, 'id'] == ['0', '2', '3', '4', 'new']


@pytest.mark.parametrize(
    'storage, config',
    [
        ('memory', None),
        ('weaviate', {'n_dim': 3, 'distance': 'l2-squared'}),
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
        ('qdrant', {'n_dim': 3, 'distance': 'euclidean'}),
        ('elasticsearch', {'n_dim': 3, 'distance': 'l2_norm'}),
        ('sqlite', dict()),
        ('redis', {'n_dim': 3, 'distance': 'L2'}),
        ('milvus', {'n_dim': 3, 'distance': 'L2'}),
        ('opensearch', {'n_dim': 3, 'distance': 'l2'}),
    ],
)
@pytest.mark.parametrize(
    'index', [1, '1', slice(1, 2), [1], [False, True, False, False, False]]
)
def test_set_and_append(index, storage, config, start_storage):
    da = DocumentArray(storage=storage, config=config)

    with da:
        da.extend([Document(id=str(i)) for i in range(5)])
    with da:
        da[index] = (
            Document(id='new')
            if isinstance(index, int) or isinstance(index, str)
            else [Document(id='new')]
        )
        da.append(Document(id='new_new'))

    assert da[:, 'id'] == ['0', 'new', '2', '3', '4', 'new_new']
