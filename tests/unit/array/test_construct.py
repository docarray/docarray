import pytest

from docarray import Document

from docarray.array.memory import DocumentArrayInMemory
from docarray.array.weaviate import DocumentArrayWeaviate, WeaviateConfig


def test_construct_docarray_weaviate(start_weaviate):
    daw = DocumentArrayWeaviate()
    daw.extend([Document(text='a'), Document(text='b'), Document(text='c')])
    name = daw.name
    del daw

    daw2 = DocumentArrayWeaviate(config=WeaviateConfig(name=name))
    assert len(daw2) == 3
    assert daw2.texts == ['a', 'b', 'c']


@pytest.mark.parametrize('da_cls', [DocumentArrayInMemory, DocumentArrayWeaviate])
def test_construct_docarray(da_cls, start_weaviate):
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

    if da_cls is DocumentArrayInMemory:
        da1 = da_cls(da)
        assert len(da1) == 10


@pytest.mark.parametrize('da_cls', [DocumentArrayInMemory])
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_singleton(da_cls, is_copy):
    d = Document()
    da = da_cls(d, copy=is_copy)
    d.id = 'hello'
    if is_copy:
        assert da[0].id != 'hello'
    else:
        assert da[0].id == 'hello'


@pytest.mark.parametrize('da_cls', [DocumentArrayInMemory])
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


@pytest.mark.parametrize('da_cls', [DocumentArrayInMemory])
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
