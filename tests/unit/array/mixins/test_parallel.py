from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import pytest

from docarray import DocumentArray, Document
from docarray.array.qdrant import DocumentArrayQdrant
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.annlite import DocumentArrayAnnlite, AnnliteConfig
from docarray.array.storage.qdrant import QdrantConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate
from docarray.array.elastic import DocumentArrayElastic, ElasticConfig
from docarray.array.redis import DocumentArrayRedis, RedisConfig
from docarray.array.milvus import DocumentArrayMilvus, MilvusConfig


def foo(d: Document):
    return (
        d.load_uri_to_image_tensor()
        .set_image_tensor_normalization()
        .set_image_tensor_channel_axis(-1, 0)
        .set_image_tensor_shape((222, 222), 0)
    )


def foo_batch(da: DocumentArray):
    for d in da:
        foo(d)
    return da


def foo_batch_with_args(da: DocumentArray, arg1, arg2):
    for d in da:
        foo(d)
    return da


@pytest.mark.parametrize('pool', [None, Pool(), ThreadPool()])
def test_parallel_map_apply_external_pool(pytestconfig, pool):
    da = DocumentArray.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')
    assert da.tensors is None
    da.apply(foo, pool=pool)
    assert da.tensors is not None


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
@pytest.mark.parametrize('backend', ['process', 'thread'])
@pytest.mark.parametrize('num_worker', [1, 2, None])
@pytest.mark.parametrize('show_progress', [True, False])
def test_parallel_map(
    pytestconfig, da_cls, config, backend, num_worker, start_storage, show_progress
):
    if __name__ == '__main__':

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        # use a generator
        for d in da.map(
            foo, backend, num_worker=num_worker, show_progress=show_progress
        ):
            assert d.tensor.shape == (3, 222, 222)

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        # use as list, here the caveat is when using process backend you can not modify thing in-place
        list(da.map(foo, backend, num_worker=num_worker))
        if backend == 'thread':
            assert da.tensors.shape == (len(da), 3, 222, 222)
        else:
            assert da.tensors is None

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]
        da_new = da.apply(foo)
        assert da_new.tensors.shape == (len(da_new), 3, 222, 222)


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
@pytest.mark.parametrize('backend', ['thread'])
@pytest.mark.parametrize('num_worker', [1, 2, None])
@pytest.mark.parametrize('b_size', [1, 2, 256])
@pytest.mark.parametrize('show_progress', [True, False])
def test_parallel_map_batch(
    pytestconfig,
    da_cls,
    config,
    backend,
    num_worker,
    b_size,
    start_storage,
    show_progress,
):
    if __name__ == '__main__':

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        # use a generator
        for _da in da.map_batch(
            foo_batch,
            batch_size=b_size,
            backend=backend,
            num_worker=num_worker,
            show_progress=True,
        ):
            for d in _da:
                assert d.tensor.shape == (3, 222, 222)

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        # use as list, here the caveat is when using process backend you can not modify thing in-place
        list(
            da.map_batch(
                foo_batch,
                batch_size=b_size,
                backend=backend,
                num_worker=num_worker,
                show_progress=True,
            )
        )
        if backend == 'thread':
            assert da.tensors.shape == (len(da), 3, 222, 222)
        else:
            assert da.tensors is None

        da_new = da.apply_batch(foo_batch, batch_size=b_size)
        assert da_new.tensors.shape == (len(da_new), 3, 222, 222)


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_map_lambda(pytestconfig, da_cls, config, start_storage):
    if __name__ == '__main__':

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        for d in da:
            assert d.tensor is None

        for d in da.map(lambda x: x.load_uri_to_image_tensor()):
            assert d.tensor is not None


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayAnnlite, AnnliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
        (DocumentArrayQdrant, QdrantConfig(n_dim=10)),
        (DocumentArrayElastic, ElasticConfig(n_dim=10)),
        (DocumentArrayRedis, RedisConfig(n_dim=10)),
        (DocumentArrayMilvus, MilvusConfig(n_dim=10)),
    ],
)
def test_apply_partial(pytestconfig, da_cls, config, start_storage):
    if __name__ == '__main__':
        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        for d in da:
            assert d.tensor is None

        da.apply_batch(partial(foo_batch_with_args, arg1=None, arg2=None), batch_size=4)

        for d in da:
            assert d.tensor is not None


@pytest.mark.parametrize(
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('annlite', AnnliteConfig(n_dim=256)),
        ('weaviate', WeaviateConfig(n_dim=256)),
        ('qdrant', QdrantConfig(n_dim=256)),
        ('elasticsearch', ElasticConfig(n_dim=256)),
        ('redis', RedisConfig(n_dim=256)),
        ('milvus', MilvusConfig(n_dim=256)),
    ],
)
@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_apply_diff_backend_storage(storage, config, backend, start_storage):
    if __name__ == '__main__':
        docs = (Document(text='hello world she smiled too much') for _ in range(1000))
        if config:
            da = DocumentArray(docs, storage=storage, config=config)
        else:
            da = DocumentArray(docs, storage=storage)

        da.apply(lambda d: d.embed_feature_hashing(), backend=backend)

        q = (
            Document(text='she smiled too much')
            .embed_feature_hashing()
            .match(da, metric='jaccard', use_scipy=True)
        )

        assert len(q.matches[:5, ('text', 'scores__jaccard__value')]) == 2
        assert len(q.matches[:5, ('text', 'scores__jaccard__value')][0]) == 5
