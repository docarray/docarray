import os
import uuid

import numpy as np
import pytest

from docarray import Document, DocumentArray
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.redis import DocumentArrayRedis, RedisConfig
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig
from docarray.helper import random_identity
from tests import random_docs


@pytest.fixture
def docs():
    return random_docs(100)


@pytest.mark.slow
@pytest.mark.parametrize('method', ['json', 'binary'])
@pytest.mark.parametrize('encoding', ['utf-8', 'cp1252'])
@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=10)),
    ],
)
def test_document_save_load(
    docs, method, encoding, tmp_path, da_cls, config, start_storage
):
    tmp_file = os.path.join(tmp_path, 'test')
    da = da_cls(docs, config=config())
    da.insert(2, Document(id='new'))
    da.save(tmp_file, file_format=method, encoding=encoding)

    da_info = {
        'id': [d.id for d in da],
        'embedding': [d.embedding for d in da],
        'content': [d.content for d in da],
    }

    if da_cls == DocumentArrayAnnlite:
        da._annlite.close()

    da_r = type(da).load(
        tmp_file, file_format=method, encoding=encoding, config=config()
    )

    assert type(da) is type(da_r)
    assert len(da) == len(da_info['id'])
    assert da_r[2].id == 'new'
    for idx, d_r in enumerate(da_r):
        assert da_info['id'][idx] == d_r.id
        np.testing.assert_equal(da_info['embedding'][idx], d_r.embedding)
        assert da_info['content'][idx] == d_r.content


@pytest.mark.parametrize('flatten_tags', [True, False])
@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=10)),
    ],
)
def test_da_csv_write(docs, flatten_tags, tmp_path, da_cls, config, start_storage):
    tmpfile = os.path.join(tmp_path, 'test.csv')
    da = da_cls(docs, config=config())
    da.save_csv(tmpfile, flatten_tags)
    with open(tmpfile) as fp:
        assert len([v for v in fp]) == len(da) + 1


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=256)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=256)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=256)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=256)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=256)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=256)),
    ],
)
def test_from_ndarray(da_cls, config, start_storage):
    _da = da_cls.from_ndarray(np.random.random([10, 256]), config=config())

    assert len(_da) == 10


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=256)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=256)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=256)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=256)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=256)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=256)),
    ],
)
def test_from_files(da_cls, config, start_storage):
    assert (
        len(
            da_cls.from_files(patterns='*.*', to_dataturi=True, size=1, config=config())
        )
        == 1
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_from_files_exclude():
    da1 = DocumentArray.from_files(f'{cur_dir}/*.*')
    has_init = False
    for s in da1[:, 'uri']:
        if s.endswith('__init__.py'):
            has_init = True
            break
    assert has_init
    da2 = DocumentArray.from_files('*.*', exclude_regex=r'__.*\.py')
    has_init = False
    for s in da2[:, 'uri']:
        if s.endswith('__init__.py'):
            has_init = True
            break
    assert not has_init


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=256)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=256)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=256)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=256)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=256)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=256)),
    ],
)
def test_from_ndjson(da_cls, config, start_storage):
    with open(os.path.join(cur_dir, 'docs.jsonlines')) as fp:
        _da = da_cls.from_ndjson(fp, config=config())
        assert len(_da) == 2


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=3)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=3)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=3)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=3)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=3)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=3)),
    ],
)
def test_from_to_pd_dataframe(da_cls, config, start_storage):
    df = da_cls.empty(2, config=config()).to_dataframe()
    assert len(da_cls.from_dataframe(df, config=config())) == 2

    # more complicated
    da = da_cls.empty(2, config=config())

    da[:, 'embedding'] = [[1, 2, 3], [4, 5, 6]]
    da[:, 'tensor'] = [[1, 2], [2, 1]]
    da[0, 'tags'] = {'hello': 'world'}
    df = da.to_dataframe()

    da2 = da_cls.from_dataframe(df, config=config())

    assert da2[0].tags == {'hello': 'world'}
    assert da2[1].tags == {}


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=3)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=3)),
        (DocumentArrayElastic, ElasticConfig(n_dim=3)),
        (DocumentArrayRedis, RedisConfig(n_dim=3)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=3)),
    ],
)
def test_from_to_bytes(da_cls, config, start_storage):
    da = da_cls.empty(2, config=config)
    bytes_data = bytes(da)

    if da_cls == DocumentArrayAnnlite:
        da._annlite.close()

    assert len(da_cls.load_binary(bytes_data)) == 2

    da = da_cls.empty(2, config=config)

    da[:, 'embedding'] = [[1, 2, 3], [4, 5, 6]]
    da[:, 'tensor'] = [[1, 2], [2, 1]]
    da[0, 'tags'] = {'hello': 'world'}

    bytes_data = bytes(da)

    if da_cls == DocumentArrayAnnlite:
        da._annlite.close()

    da2 = da_cls.load_binary(bytes_data)
    assert da2.tensors == [[1, 2], [2, 1]]
    import numpy as np

    np.testing.assert_array_equal(da2[:, 'embedding'], [[1, 2, 3], [4, 5, 6]])
    # assert da2.embeddings == [[1, 2, 3], [4, 5, 6]]
    assert da2[0].tags == {'hello': 'world'}
    assert da2[1].tags == {}


@pytest.mark.parametrize('show_progress', [True, False])
@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArrayInMemory, lambda: None),
        (DocumentArraySqlite, lambda: None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=256)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=256)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=256)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=256)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=256)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=256)),
    ],
)
def test_push_pull_io(da_cls, config, show_progress, start_storage):
    da1 = da_cls.empty(10, config=config())

    da1[:, 'embedding'] = np.random.random([len(da1), 256])
    random_texts = [str(uuid.uuid1()) for _ in da1]
    da1[:, 'text'] = random_texts

    name = f'docarray_ci_{random_identity()}'

    da1.push(name, show_progress=show_progress)

    da2 = da_cls.pull(name, show_progress=show_progress, config=config())

    assert len(da1) == len(da2) == 10
    assert da1[:, 'text'] == da2[:, 'text'] == random_texts

    all_names = DocumentArray.cloud_list()

    assert name in all_names

    DocumentArray.cloud_delete(name)

    all_names = DocumentArray.cloud_list()

    assert name not in all_names


@pytest.mark.parametrize(
    'protocol', ['protobuf', 'pickle', 'protobuf-array', 'pickle-array']
)
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=3)),
        # (DocumentArrayAnnlite, PqliteConfig(n_dim=3)), # TODO: enable this
        # (DocumentArrayQdrant, QdrantConfig(n_dim=3)),
        # (DocumentArrayElastic, ElasticConfig(n_dim=3)), # Elastic needs config
        # (DocumentArrayRedis, RedisConfig(n_dim=3)), # Redis needs config
        # (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=3)),
    ],
)
def test_from_to_base64(protocol, compress, da_cls, config):
    da = da_cls.empty(10, config=config)

    da[:, 'embedding'] = [[1, 2, 3]] * len(da)
    da_r = da_cls.from_base64(da.to_base64(protocol, compress), protocol, compress)

    # only pickle-array will serialize the configuration so we can assume DAs are equal
    if protocol == 'pickle-array':
        assert da_r == da
    # for the rest, we can only check the docs content
    else:
        for d1, d2 in zip(da_r, da):
            assert d1 == d2
    # assert da_r[0].embedding == [1, 2, 3]
    np.testing.assert_array_equal(da_r[0].embedding, [1, 2, 3])
