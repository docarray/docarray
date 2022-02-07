import pytest

from docarray import DocumentArray, Document
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.storage.weaviate import WeaviateConfig
from docarray.array.weaviate import DocumentArrayWeaviate

N = 100


def da_and_dam():
    da = DocumentArray.empty(N)
    dasq = DocumentArraySqlite.empty(N)
    return (da, dasq)


@pytest.fixture
def docs():
    yield (Document(text=str(j)) for j in range(100))


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
    ],
)
def test_iter_len_bool(da_cls, config, start_weaviate):
    if config:
        da = da_cls.empty(N, config=config)
    else:
        da = da_cls.empty(N)
    j = 0
    for _ in da:
        j += 1
    assert j == N
    assert j == len(da)
    assert da
    da.clear()
    assert not da


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
    ],
)
def test_repr(da_cls, config, start_weaviate):
    if config:
        da = da_cls.empty(N, config=config)
    else:
        da = da_cls.empty(N)
    assert f'length={N}' in repr(da)


@pytest.mark.parametrize(
    'storage, config',
    [('memory', None), ('sqlite', None), ('weaviate', WeaviateConfig(n_dim=128))],
)
def test_repr_str(docs, storage, config, start_weaviate):
    if config:
        da = DocumentArray(docs, storage=storage, config=config)
    else:
        da = DocumentArray(docs, storage=storage)
    da.summary()
    assert da
    da.clear()
    assert not da
    print(da)


@pytest.mark.parametrize(
    'da_cls, config',
    [
        (DocumentArray, None),
        (DocumentArraySqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=10)),
    ],
)
def test_iadd(da_cls, config, start_weaviate):
    if config:
        da = da_cls.empty(N, config=config)
    else:
        da = da_cls.empty(N)
    oid = id(da)
    dap = DocumentArray.empty(10)
    da += dap
    assert len(da) == N + len(dap)
    nid = id(da)
    assert nid == oid


@pytest.mark.parametrize('da', [da_and_dam()[0]])
def test_add(da):
    oid = id(da)
    dap = DocumentArray.empty(10)
    da = da + dap
    assert len(da) == N + len(dap)
    nid = id(da)
    assert nid != oid
