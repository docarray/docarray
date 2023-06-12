import numpy as np
import pytest

from docarray import BaseDoc, DocDict, DocList, DocVec
from docarray.typing import NdArray


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
    docs = DocDict[MyDoc].from_docs(
        DocList([MyDoc(id='a', text='a'), MyDoc(id='b', text='b')])
    )
    assert list(docs) == ['a', 'b']


def test_init_from_doc_list_raw():
    docs = DocDict[MyDoc].from_docs([MyDoc(id='a', text='a'), MyDoc(id='b', text='b')])
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


def test_update_iterable(docs):
    docs.update([MyDoc(id='c', text='c')])
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


class InerDoc(BaseDoc):
    text: str


class NestedDoc(BaseDoc):
    doc: InerDoc


@pytest.fixture
def docs_nested():
    return DocDict[NestedDoc](
        x=NestedDoc(id='a', doc=InerDoc(id='a', text='a')),
        y=NestedDoc(id='b', doc=InerDoc(id='b', text='b')),
    )


def test_getatr_nested(docs_nested):
    assert docs_nested.id == {'x': 'a', 'y': 'b'}

    nested = docs_nested.doc
    assert isinstance(nested, DocDict[InerDoc])
    assert nested.text == {'x': 'a', 'y': 'b'}


def test_setattr_nested(docs_nested):
    docs_nested.doc = {'x': InerDoc(id='c', text='c'), 'y': InerDoc(id='d', text='d')}
    assert docs_nested.doc.text == {'x': 'c', 'y': 'd'}


def test_to_doc_list(docs):
    assert docs.to_doc_list() == DocList(
        [MyDoc(id='a', text='a'), MyDoc(id='b', text='b')]
    )


def test_to_doc_vec():
    class MyDoc(BaseDoc):
        tensor: NdArray[2, 3]

    docs = DocDict[MyDoc](
        a=MyDoc(id='a', tensor=np.array([[1, 2, 3], [4, 5, 6]])),
        b=MyDoc(id='b', tensor=np.array([[7, 8, 9], [10, 11, 12]])),
    )

    docs2 = docs.to_doc_vec()

    assert isinstance(docs2, DocVec)
    assert isinstance(docs2, DocVec[MyDoc])

    assert docs2.tensor.shape == (2, 2, 3)

    assert (docs2[0].tensor == docs2[0].tensor).all()
