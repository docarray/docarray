from typing import Dict, List, Optional, Set

import pytest

from docarray import BaseDoc, DocArray
from docarray.documents import ImageDoc


class InnerDoc(BaseDoc):
    integer: int
    inner_list: List


class MMDoc(BaseDoc):
    text: str = ''
    price: int = 0
    categories: Optional[List[str]] = None
    image: Optional[ImageDoc] = None
    matches: Optional[DocArray] = None
    matches_with_same_id: Optional[DocArray] = None
    opt_int: Optional[int] = None
    test_set: Optional[Set] = None
    inner_doc: Optional[InnerDoc] = None
    test_dict: Optional[Dict] = None


@pytest.fixture
def doc1():
    return MMDoc(
        text='hey here',
        categories=['a', 'b', 'c'],
        price=10,
        matches=DocArray[MMDoc]([MMDoc()]),
        matches_with_same_id=DocArray[MMDoc](
            [MMDoc(id='a', matches=DocArray[MMDoc]([MMDoc()]))]
        ),
        test_set={'a', 'a'},
        inner_doc=InnerDoc(integer=2, inner_list=['c', 'd']),
        test_dict={'a': 0, 'b': 2, 'd': 4, 'z': 3},
    )


@pytest.fixture
def doc2(doc1):
    return MMDoc(
        id=doc1.id,
        text='hey here 2',
        categories=['d', 'e', 'f'],
        price=5,
        opt_int=5,
        matches=DocArray[MMDoc]([MMDoc()]),
        matches_with_same_id=DocArray[MMDoc](
            [MMDoc(id='a', matches=DocArray[MMDoc]([MMDoc()]))]
        ),
        test_set={'a', 'b'},
        inner_doc=InnerDoc(integer=3, inner_list=['a', 'b']),
        test_dict={'a': 10, 'b': 10, 'c': 3, 'z': None},
    )


def test_update_complex(doc1, doc2):
    doc1.update(doc2)
    # doc1 is changed in place (no extra memory)
    assert doc1.text == 'hey here 2'
    assert doc1.categories == ['a', 'b', 'c', 'd', 'e', 'f']
    assert len(doc1.matches) == 2
    assert doc1.opt_int == 5
    assert doc1.price == 5
    assert doc1.test_set == {'a', 'b'}
    assert len(doc1.matches_with_same_id) == 1
    assert len(doc1.matches_with_same_id[0].matches) == 2
    assert doc1.inner_doc.integer == 3
    assert doc1.inner_doc.inner_list == ['c', 'd', 'a', 'b']
    assert doc1.test_dict == {'a': 10, 'b': 10, 'c': 3, 'd': 4, 'z': None}


def test_update_simple():
    class MyDocument(BaseDoc):
        content: str
        title: Optional[str] = None
        tags_: List

    my_doc1 = MyDocument(
        content='Core content of the document', title='Title', tags_=['python', 'AI']
    )
    my_doc2 = MyDocument(content='Core content updated', tags_=['docarray'])

    my_doc1.update(my_doc2)
    assert my_doc1.content == 'Core content updated'
    assert my_doc1.title == 'Title'
    assert my_doc1.tags_ == ['python', 'AI', 'docarray']


def test_update_different_schema_fails():
    class DocA(BaseDoc):
        content: str

    class DocB(BaseDoc):
        image: Optional[ImageDoc] = None

    docA = DocA(content='haha')
    docB = DocB()
    with pytest.raises(Exception):
        docA.update(docB)
