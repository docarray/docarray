import os

import numpy as np
import pytest

from docarray import DocumentArray, Document
from docarray.array.annlite import DocumentArrayAnnlite
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.storage.annlite import AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig


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

    for doc in daq:
        doc.embedding = np.random.random(128)

    for doc in dai:
        doc.embedding = np.random.random(128)

    return daq, dai


def test_empty_doc(embed_docs):
    da = DocumentArray([Document(embedding=np.random.random(128))])
    with pytest.raises(ValueError):
        da[0].plot_matches_sprites()

    daq, dai = embed_docs

    with pytest.raises(ValueError):
        daq[0].plot_matches_sprites()

    with pytest.raises(ValueError):
        daq[0].plot_matches_sprites(top_k=0)


@pytest.mark.parametrize('top_k', [1, 10, 20])
@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=128)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=128, scroll_batch_size=8)),
        (DocumentArrayElastic, ElasticConfig(n_dim=128)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=128)),
    ],
)
def test_matches_sprites(
    pytestconfig, tmpdir, da_cls, config, embed_docs, start_storage, top_k
):
    da, das = embed_docs
    if config:
        das = da_cls(das, config=config)
    else:
        das = da_cls(das)
    da.match(das)
    da[0].plot_matches_sprites(top_k, output=tmpdir / 'sprint_da.png')
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
        (DocumentArrayMilvus, lambda: MilvusConfig(n_dim=128)),
    ],
)
def test_matches_sprite_image_generator(
    pytestconfig,
    tmpdir,
    image_source,
    da_cls,
    config_gen,
    embed_docs,
    start_storage,
):
    da, das = embed_docs
    if (
        image_source == 'tensor' and da_cls != DocumentArrayMilvus
    ):  # Milvus can't handle large tensors
        da.apply(lambda d: d.load_uri_to_image_tensor())
        das.apply(lambda d: d.load_uri_to_image_tensor())

    if config_gen:
        das = da_cls(das, config=config_gen())
    else:
        das = da_cls(das)
    da.match(das)
    da[0].plot_matches_sprites(output=tmpdir / 'sprint_da.png')
    assert os.path.exists(tmpdir / 'sprint_da.png')
