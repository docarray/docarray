import pytest

from docarray import BaseDoc, DocDict, DocList


class MyDoc(BaseDoc):
    text: str


def test_init():
    docs = DocDict[MyDoc](x=MyDoc(id='a', text='a'), y=MyDoc(id='b', text='b'))
    assert list(docs) == ['x', 'y']


def test_init_validate_fail():
    with pytest.raises(ValueError):
        DocDict[MyDoc](x='hello')


def test_init_raw():
    docs = DocDict(x=MyDoc(id='a', text='a'), y=MyDoc(id='b', text='b'))
    assert list(docs) == ['x', 'y']


def test_init_from_doc_list():
    docs = DocDict[MyDoc].from_doc_list(
        DocList([MyDoc(id='a', text='a'), MyDoc(id='b', text='b')])
    )
    assert list(docs) == ['a', 'b']


def test_init_from_doc_list_raw():
    docs = DocDict[MyDoc].from_doc_list(
        DocList([MyDoc(id='a', text='a'), MyDoc(id='b', text='b')])
    )
    assert list(docs) == ['a', 'b']


@pytest.fixture
def docs():
    return DocDict[MyDoc](x=MyDoc(id='a', text='a'), y=MyDoc(id='b', text='b'))


def test_update(docs):
    docs.update({'z': MyDoc(id='c', text='c')})
    assert list(docs) == ['x', 'y', 'z']


def test_update_doc_dict(docs):
    docs.update(DocDict(z=MyDoc(id='c', text='c')))
    assert list(docs) == ['x', 'y', 'z']


def test_update_doc_list(docs):
    docs.update(DocList([MyDoc(id='c', text='c')]))
    assert list(docs) == ['x', 'y', 'c']


def test_getitem(docs):
    assert docs['x'] == MyDoc(id='a', text='a')
    assert docs['y'] == MyDoc(id='b', text='b')


def test_getatr(docs):
    assert docs.text == {'x': 'a', 'y': 'b'}
    assert docs.id == {'x': 'a', 'y': 'b'}


def test_setattr(docs):
    docs.text = {'x': 'c', 'y': 'd'}
    assert docs.text == {'x': 'c', 'y': 'd'}

    assert docs['x'].text == 'c'
