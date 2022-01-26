import pytest

from docarray import DocumentArray, Document
from docarray.array.sqlite import DocumentArraySqlite

N = 100


def da_and_dam():
    da = DocumentArray.empty(N)
    dasq = DocumentArraySqlite.empty(N)
    return (da, dasq)


@pytest.fixture
def docs():
    yield (Document(text=str(j)) for j in range(100))


@pytest.mark.parametrize('da', da_and_dam())
def test_iter_len_bool(da):
    j = 0
    for _ in da:
        j += 1
    assert j == N
    assert j == len(da)
    assert da
    da.clear()
    assert not da


@pytest.mark.parametrize('da', da_and_dam())
def test_repr(da):
    assert f'length={N}' in repr(da)


@pytest.mark.parametrize('storage', ['memory', 'sqlite'])
def test_repr_str(docs, storage):
    da = DocumentArray(docs, storage=storage)
    print(da)
    da.summary()
    assert da
    da.clear()
    assert not da
    print(da)


@pytest.mark.parametrize('da', da_and_dam())
def test_iadd(da):
    oid = id(da)
    dap = DocumentArray.empty(10)
    da += dap
    assert len(da) == N + len(dap)
    nid = id(da)
    assert nid == oid


@pytest.mark.parametrize('da', da_and_dam())
def test_add(da):
    oid = id(da)
    dap = DocumentArray.empty(10)
    da = da + dap
    assert len(da) == N + len(dap)
    nid = id(da)
    assert nid != oid
