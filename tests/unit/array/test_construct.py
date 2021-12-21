import pytest

from docarray import Document, DocumentArray


@pytest.mark.parametrize('da_cls', [DocumentArray])
def test_construct_docarray(da_cls):
    da = da_cls()
    assert len(da) == 0

    da = da_cls(Document())
    assert len(da) == 1

    da = da_cls([Document(), Document()])
    assert len(da) == 2

    da = da_cls((Document(), Document()))
    assert len(da) == 2

    da = da_cls((Document() for _ in range(10)))
    assert len(da) == 10

    da1 = da_cls(da)
    assert len(da1) == 10


@pytest.mark.parametrize('da_cls', [DocumentArray])
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_singleton(da_cls, is_copy):
    d = Document()
    da = da_cls(d, copy=is_copy)
    d.id = 'hello'
    if is_copy:
        assert da[0].id != 'hello'
    else:
        assert da[0].id == 'hello'


@pytest.mark.parametrize('da_cls', [DocumentArray])
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_da(da_cls, is_copy):
    d1 = Document()
    d2 = Document()
    da = da_cls([d1, d2], copy=is_copy)
    d1.id = 'hello'
    if is_copy:
        assert da[0].id != 'hello'
    else:
        assert da[0].id == 'hello'


@pytest.mark.parametrize('da_cls', [DocumentArray])
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_list(da_cls, is_copy):
    d1 = Document()
    d2 = Document()
    da = da_cls([d1, d2], copy=is_copy)
    d1.id = 'hello'
    if is_copy:
        assert da[0].id != 'hello'
    else:
        assert da[0].id == 'hello'
