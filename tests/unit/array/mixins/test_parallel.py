import pytest

from docarray import DocumentArray, Document
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.pqlite import DocumentArrayPqlite, PqliteConfig
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate


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


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayPqlite, PqliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
    ],
)
@pytest.mark.parametrize('backend', ['process', 'thread'])
@pytest.mark.parametrize('num_worker', [1, 2, None])
def test_parallel_map(
    pytestconfig, da_cls, config, backend, num_worker, start_weaviate
):
    if __name__ == '__main__':

        if config:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg', config=config)[
                :10
            ]
        else:
            da = da_cls.from_files(f'{pytestconfig.rootdir}/**/*.jpeg')[:10]

        # use a generator
        for d in da.map(foo, backend, num_worker=num_worker):
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
        (DocumentArrayPqlite, PqliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
    ],
)
@pytest.mark.parametrize('backend', ['thread'])
@pytest.mark.parametrize('num_worker', [1, 2, None])
@pytest.mark.parametrize('b_size', [1, 2, 256])
def test_parallel_map_batch(
    pytestconfig, da_cls, config, backend, num_worker, b_size, start_weaviate
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
            foo_batch, batch_size=b_size, backend=backend, num_worker=num_worker
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
                foo_batch, batch_size=b_size, backend=backend, num_worker=num_worker
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
        (DocumentArrayPqlite, PqliteConfig(n_dim=10)),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
    ],
)
def test_map_lambda(pytestconfig, da_cls, config, start_weaviate):
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
    'storage,config',
    [
        ('memory', None),
        ('sqlite', None),
        ('pqlite', PqliteConfig(n_dim=256)),
        ('weaviate', WeaviateConfig(n_dim=256)),
    ],
)
@pytest.mark.parametrize('backend', ['thread', 'process'])
def test_apply_diff_backend_storage(storage, config, backend, start_weaviate):
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
