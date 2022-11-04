import json
import os
import random

import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.annlite import DocumentArrayAnnlite
from docarray.array.storage.annlite import AnnliteConfig
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.redis import DocumentArrayRedis, RedisConfig
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig


@pytest.mark.parametrize('keep_aspect_ratio', [True, False])
@pytest.mark.parametrize('show_index', [True, False])
@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        # (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128, scroll_batch_size=8)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, RedisConfig(n_dim=128)),
        # (DocumentArrayMilvus, MilvusConfig(n_dim=128)),  # tensor is too large to handle
    ],
)
def test_sprite_fail_tensor_success_uri(
    pytestconfig, tmpdir, da_cls, config, start_storage, keep_aspect_ratio, show_index
):
    files = [
        f'{pytestconfig.rootdir}/tests/image-data/*.jpg',
        f'{pytestconfig.rootdir}/tests/image-data/*.png',
    ]
    if config:
        da = da_cls.from_files(files, config=config)
    else:
        da = da_cls.from_files(files)
    da.apply(
        lambda d: d.load_uri_to_image_tensor().set_image_tensor_channel_axis(-1, 0)
    )
    with pytest.raises(ValueError):
        da.plot_image_sprites()
    da.plot_image_sprites(
        tmpdir / 'sprint_da.png',
        image_source='uri',
        keep_aspect_ratio=keep_aspect_ratio,
        show_index=show_index,
    )
    da.save_gif(tmpdir / 'sprint_da.gif', show_index=show_index, channel_axis=0)
    assert os.path.exists(tmpdir / 'sprint_da.png')


@pytest.mark.parametrize('image_source', ['tensor', 'uri'])
@pytest.mark.parametrize(
    'da_cls,config_gen',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=128, scroll_batch_size=8)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=128)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=128)),
        # (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=128)),
    ],
)
@pytest.mark.parametrize('canvas_size', [50, 512])
@pytest.mark.parametrize('min_size', [16, 64])
def test_sprite_image_generator(
    pytestconfig,
    tmpdir,
    image_source,
    da_cls,
    config_gen,
    canvas_size,
    min_size,
    start_storage,
):
    files = [
        f'{pytestconfig.rootdir}/tests/image-data/*.jpg',
        f'{pytestconfig.rootdir}/tests/image-data/*.png',
    ]
    if config_gen:
        da = da_cls.from_files(files, config=config_gen())
    else:
        da = da_cls.from_files(files)
    da.apply(lambda d: d.load_uri_to_image_tensor())
    da.plot_image_sprites(
        tmpdir / 'sprint_da.png',
        image_source=image_source,
        canvas_size=canvas_size,
        min_size=min_size,
    )
    assert os.path.exists(tmpdir / 'sprint_da.png')


@pytest.fixture
def da_and_dam(start_storage):
    embeddings = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    return [
        cls(
            [
                Document(embedding=x, tags={'label': random.randint(0, 5)})
                for x in embeddings
            ],
            **kwargs,
        )
        for cls, kwargs in [
            (DocumentArray, {}),
            (DocumentArraySqlite, {}),
            (DocumentArrayWeaviate, {'config': {'n_dim': 3}}),
            (DocumentArrayAnnlite, {'config': {'n_dim': 3}}),
            (DocumentArrayQdrant, {'config': {'n_dim': 3}}),
            (DocumentArrayRedis, {'config': {'n_dim': 3}}),
            (DocumentArrayMilvus, {'config': {'n_dim': 3}}),
        ]
    ]


def test_plot_embeddings(da_and_dam):
    for da in da_and_dam:
        _test_plot_embeddings(da)


def test_plot_sprites(tmpdir):
    da = DocumentArray.empty(5)
    da.tensors = np.random.random([5, 3, 226, 226])
    da.plot_image_sprites(tmpdir / 'a.png', channel_axis=0, show_index=True)
    assert os.path.exists(tmpdir / 'a.png')


def _test_plot_embeddings(da):
    with da:
        p = da.plot_embeddings(start_server=False)
    assert os.path.exists(p)
    assert os.path.exists(os.path.join(p, 'config.json'))
    with open(os.path.join(p, 'config.json')) as fp:
        config = json.load(fp)
        assert len(config['embeddings']) == 1
        assert config['embeddings'][0]['tensorShape'] == list(da.embeddings.shape)


@pytest.mark.parametrize(
    'da_cls,config_gen',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=5)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=5)),
        (DocumentArrayQdrant, lambda: QdrantConfig(n_dim=5)),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=5)),
        (DocumentArrayRedis, lambda: RedisConfig(n_dim=5)),
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=5)),
    ],
)
def test_plot_embeddings_same_path(tmpdir, da_cls, config_gen, start_storage):
    if config_gen:
        da1 = da_cls.empty(100, config=config_gen())
        da2 = da_cls.empty(768, config=config_gen())
    else:
        da1 = da_cls.empty(100)
        da2 = da_cls.empty(768)
    with da1:
        da1.embeddings = np.random.random([100, 5])
        p1 = da1.plot_embeddings(start_server=False, path=tmpdir)
    with da2:
        da2.embeddings = np.random.random([768, 5])
        p2 = da2.plot_embeddings(start_server=False, path=tmpdir)
    assert p1 == p2
    assert os.path.exists(p1)
    with open(os.path.join(p1, 'config.json')) as fp:
        config = json.load(fp)
        assert len(config['embeddings']) == 2


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
def test_summary_homo_hetero(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(100, config=config)
    else:
        da = da_cls.empty(100)
    with da:
        da._get_attributes()
        da.summary()
        da._get_raw_summary()

    da[0].pop('id')
    with da:
        da.summary()

        da._get_raw_summary()


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
def test_empty_get_attributes(da_cls, config, start_storage):
    if config:
        da = da_cls.empty(10, config=config)
    else:
        da = da_cls.empty(10)
    da[0].pop('id')
    print(da[:, 'id'])
