import pytest

from docarray3 import Document, DocumentArray


@pytest.mark.parametrize('da_cls', [DocumentArray])
def test_insert(da_cls):
    da = da_cls()
    da.insert(0, Document(text='hello'))
    da.insert(0, Document(text='world'))
    assert da[0].text == 'world'
    assert da[1].text == 'hello'


@pytest.mark.parametrize('da_cls', [DocumentArray])
def test_length(da_cls):
    da = da_cls()
    assert len(da)
    da.insert(0, Document(text='hello'))
    da.insert(0, Document(text='world'))
    assert da[0].text == 'world'
    assert da[1].text == 'hello'
