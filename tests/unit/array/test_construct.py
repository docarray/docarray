import pytest

from docarray import Document
from docarray.array.memory import DocumentArrayInMemory
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.pqlite import DocumentArrayPqlite
from docarray.array.weaviate import DocumentArrayWeaviate, WeaviateConfig


def test_construct_docarray_weaviate(start_weaviate):
    daw = DocumentArrayWeaviate(config=WeaviateConfig(n_dim=10))
    daw.extend([Document(text='a'), Document(text='b'), Document(text='c')])
    name = daw.name
    del daw

    daw2 = DocumentArrayWeaviate(config=WeaviateConfig(n_dim=10, name=name))
    assert len(daw2) == 3
    assert daw2.texts == ['a', 'b', 'c']


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayPqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
    ],
)
def test_construct_docarray(da_cls, config, start_weaviate):
    if config:
        da = da_cls(config=config)
        assert len(da) == 0

        da = da_cls(Document(), config=config)
        assert len(da) == 1

        da = da_cls([Document(), Document()], config=config)
        assert len(da) == 2

        da = da_cls((Document(), Document()), config=config)
        assert len(da) == 2

        da = da_cls((Document() for _ in range(10)), config=config)
        assert len(da) == 10
    else:
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


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayPqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
    ],
)
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_singleton(da_cls, config, is_copy, start_weaviate):
    d = Document()
    if config:
        da = da_cls(d, copy=is_copy, config=config)
    else:
        da = da_cls(d, copy=is_copy)

    d.id = 'hello'
    if da_cls == DocumentArrayInMemory:
        if is_copy:
            assert da[0].id != 'hello'
        else:
            assert da[0].id == 'hello'
    else:
        assert da[0].id != 'hello'


@pytest.mark.parametrize(
    'da_cls,config',
    [
        (DocumentArrayInMemory, None),
        (DocumentArraySqlite, None),
        (DocumentArrayPqlite, None),
        (DocumentArrayWeaviate, WeaviateConfig(n_dim=128)),
    ],
)
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_da(da_cls, config, is_copy, start_weaviate):
    d1 = Document()
    d2 = Document()
    if config:
        da = da_cls([d1, d2], copy=is_copy, config=config)
    else:
        da = da_cls([d1, d2], copy=is_copy)
    d1.id = 'hello'
    if da_cls == DocumentArrayInMemory:
        if is_copy:
            assert da[0].id != 'hello'
        else:
            assert da[0].id == 'hello'
    else:
        assert da[0] != 'hello'


@pytest.mark.parametrize(
    'da_cls', [DocumentArrayInMemory, DocumentArraySqlite, DocumentArrayPqlite]
)
@pytest.mark.parametrize('is_copy', [True, False])
def test_docarray_copy_list(da_cls, is_copy, start_weaviate):
    d1 = Document()
    d2 = Document()
    da = da_cls([d1, d2], copy=is_copy)
    d1.id = 'hello'
    if da_cls == DocumentArrayInMemory:
        if is_copy:
            assert da[0].id != 'hello'
        else:
            assert da[0].id == 'hello'
    else:
        assert da[0] != 'hello'
