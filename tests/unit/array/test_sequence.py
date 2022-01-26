import pytest

from docarray import Document, DocumentArray


@pytest.mark.parametrize('da_cls', [DocumentArray])
def test_insert(da_cls):
    da = da_cls()
    assert not len(da)
    da.insert(0, Document(text='hello', id="0"))
    da.insert(0, Document(text='world', id="1"))
    assert len(da) == 2
    assert da[0].text == 'world'
    assert da[1].text == 'hello'
    assert da["1"].text == 'world'
    assert da["0"].text == 'hello'


@pytest.mark.parametrize('da_cls', [DocumentArray])
def test_append_extend(da_cls):
    da = da_cls()
    da.append(Document())
    da.append(Document())
    assert len(da) == 2
    da.extend([Document(), Document()])
    assert len(da) == 4
