import numpy as np
import pytest

from docarray import DocumentArray
from docarray.array.sqlite import DocumentArraySqlite
from docarray.array.weaviate import DocumentArrayWeaviate


@pytest.mark.parametrize('cls', [DocumentArray, DocumentArraySqlite])
@pytest.mark.parametrize(
    'content_attr', ['texts', 'embeddings', 'tensors', 'blobs', 'contents']
)
def test_content_empty_getter_return_none(cls, content_attr):
    da = cls()
    assert getattr(da, content_attr) is None


@pytest.mark.parametrize('cls', [DocumentArray, DocumentArraySqlite])
@pytest.mark.parametrize(
    'content_attr',
    [
        ('texts', ''),
        ('embeddings', np.array([])),
        ('tensors', np.array([])),
        ('blobs', []),
        ('contents', []),
    ],
)
def test_content_empty_setter(cls, content_attr):
    da = cls()
    setattr(da, content_attr[0], content_attr[1])
    assert getattr(da, content_attr[0]) is None


@pytest.mark.parametrize(
    'cls', [DocumentArray, DocumentArraySqlite, DocumentArrayWeaviate]
)
@pytest.mark.parametrize(
    'content_attr',
    [
        ('texts', ['s'] * 10),
        ('tensors', np.random.random([10, 2])),
        ('blobs', [b's'] * 10),
    ],
)
def test_content_getter_setter(cls, content_attr, start_storage):
    da = cls.empty(10)
    setattr(da, content_attr[0], content_attr[1])
    np.testing.assert_equal(da.contents, content_attr[1])
    da.contents = content_attr[1]
    np.testing.assert_equal(da.contents, content_attr[1])
    np.testing.assert_equal(getattr(da, content_attr[0]), content_attr[1])
    da.contents = None
    assert da.contents is None


@pytest.mark.parametrize('da_len', [0, 1, 2])
@pytest.mark.parametrize(
    'cls', [DocumentArray, DocumentArraySqlite, DocumentArrayWeaviate]
)
def test_content_empty(da_len, cls, start_storage):
    da = cls.empty(da_len)
    assert not da.texts
    assert not da.contents
    assert not da.tensors
    assert not da.blobs

    da.texts = ['hello'] * da_len
    if da_len == 0:
        assert not da.contents
    else:
        assert da.contents == ['hello'] * da_len
        assert da.texts == ['hello'] * da_len
        assert not da.tensors
        assert not da.blobs


@pytest.mark.parametrize('da_len', [0, 1, 2])
@pytest.mark.parametrize(
    'cls', [DocumentArray, DocumentArraySqlite, DocumentArrayWeaviate]
)
def test_embeddings_setter(da_len, cls, start_storage):
    da = cls.empty(da_len)
    da.embeddings = np.random.rand(da_len, 5)
    for doc in da:
        assert doc.embedding.shape == (5,)
