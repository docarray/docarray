import numpy as np
import pytest

from docarray import DocumentArray


@pytest.mark.parametrize('cls', [DocumentArray])
@pytest.mark.parametrize(
    'content_attr', ['texts', 'embeddings', 'blobs', 'buffers', 'contents']
)
def test_content_empty_getter_return_none(cls, content_attr):
    da = cls()
    assert getattr(da, content_attr) is None


@pytest.mark.parametrize('cls', [DocumentArray])
@pytest.mark.parametrize(
    'content_attr',
    [
        ('texts', ''),
        ('embeddings', np.array([])),
        ('blobs', np.array([])),
        ('buffers', []),
        ('contents', []),
    ],
)
def test_content_empty_setter(cls, content_attr):
    da = cls()
    setattr(da, content_attr[0], content_attr[1])
    assert getattr(da, content_attr[0]) is None


@pytest.mark.parametrize('cls', [DocumentArray])
@pytest.mark.parametrize(
    'content_attr',
    [
        ('texts', ['s'] * 10),
        ('blobs', np.random.random([10, 2])),
        ('buffers', [b's'] * 10),
    ],
)
def test_content_getter_setter(cls, content_attr):
    da = cls.empty(10)
    setattr(da, content_attr[0], content_attr[1])
    np.testing.assert_equal(da.contents, content_attr[1])
    da.contents = content_attr[1]
    np.testing.assert_equal(da.contents, content_attr[1])
    np.testing.assert_equal(getattr(da, content_attr[0]), content_attr[1])
    da.contents = None
    assert da.contents is None


@pytest.mark.parametrize('da_len', [0, 1, 2])
def test_content_empty(da_len):
    da = DocumentArray.empty(da_len)
    assert not da.texts
    assert not da.contents
    assert not da.blobs
    assert not da.buffers

    da.texts = ['hello'] * da_len
    if da_len == 0:
        assert not da.contents
    else:
        assert da.contents == ['hello'] * da_len
        assert da.texts == ['hello'] * da_len
        assert not da.blobs
        assert not da.buffers
