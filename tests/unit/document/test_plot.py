import json
import os
import random

import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.annlite import DocumentArrayAnnlite
from docarray.array.storage.annlite import AnnliteConfig
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig


@pytest.fixture()
def embed_docs(pytestconfig):
    index_files = [
        f'{pytestconfig.rootdir}/tests/image-data/*.jpg',
    ]
    query_file = [
        f'{pytestconfig.rootdir}/tests/image-data/*.png',
    ]
    dai = DocumentArray.from_files(index_files)
    daq = DocumentArray.from_files(query_file)

    for doc in dai + daq:
        doc.embedding = np.random.random(128)

    return daq, dai


@pytest.mark.parametrize('top_k', [1, 5, 10, 15, 20])
@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128, scroll_batch_size=8)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
    ],
)
def test_matching_sprites(
    pytestconfig, tmpdir, da_cls, config, embed_docs, start_storage, top_k
):
    da, das = embed_docs
    if config:
        das = da_cls(das, config=config)
    else:
        das = da_cls(das)
    da.match(das, limit=10)
    da[0].plot_matching_sprites(
        tmpdir / 'sprint_da.png', image_source='uri', top_k=top_k
    )
    assert os.path.exists(tmpdir / 'sprint_da.png')


@pytest.mark.parametrize('image_source', ['tensor', 'uri'])
@pytest.mark.parametrize(
    'da_cls,config_gen',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, lambda: AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, lambda: WeaviateConfig(n_dim=128)),
        (
            DocumentArrayQdrant,
            lambda: QdrantConfig(n_dim=128, scroll_batch_size=8),
        ),
        (DocumentArrayElastic, lambda: ElasticConfig(n_dim=128)),
    ],
)
def test_matching_sprite_image_generator(
    pytestconfig,
    tmpdir,
    image_source,
    da_cls,
    config_gen,
    embed_docs,
    start_storage,
):
    da, das = embed_docs
    if config_gen:
        da = da_cls(das, config=config_gen)
    else:
        da = da_cls(das)
    da.match(das, limit=10)
    da.apply(lambda d: d.load_uri_to_image_tensor())
    da[0].plot_matching_sprites(tmpdir / 'sprint_da.png', image_source=image_source)
    assert os.path.exists(tmpdir / 'sprint_da.png')
