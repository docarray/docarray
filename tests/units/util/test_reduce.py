import pytest
from typing import Optional, List, Dict, Any, Set
from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.utils.reduce import reduce_docs, reduce, reduce_all


class InnerDoc(BaseDocument):
    integer: int
    l: List


class MMDoc(BaseDocument):
    text: str = ''
    price: int = 0
    categories: Optional[List[str]] = None
    image: Optional[Image] = None
    matches: Optional[DocumentArray] = None
    matches_with_same_id: Optional[DocumentArray] = None
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
        matches=DocumentArray[MMDoc]([MMDoc()]),
        matches_with_same_id=DocumentArray[MMDoc](
            [MMDoc(id='a', matches=DocumentArray[MMDoc]([MMDoc()]))]
        ),
        test_set={'a', 'a'},
        inner_doc=InnerDoc(integer=2, l=['c', 'd']),
        test_dict={'a': 0, 'b': 2, 'd': 4}
    )


@pytest.fixture
def doc2(doc1):
    return MMDoc(
        id=doc1.id,
        text='hey here 2',
        categories=['d', 'e', 'f'],
        price=5,
        opt_int=5,
        matches=DocumentArray[MMDoc]([MMDoc()]),
        matches_with_same_id=DocumentArray[MMDoc](
            [MMDoc(id='a', matches=DocumentArray[MMDoc]([MMDoc()]))]
        ),
        test_set={'a', 'b'},
        inner_doc=InnerDoc(integer=3, l=['a', 'b']),
        test_dict={'a': 10, 'b': 10, 'c': 3}
    )


def test_reduce_docs(doc1, doc2):
    result = reduce_docs(doc1, doc2)
    assert result.text == 'hey here'
    assert len(result.matches) == 2
    assert result.categories == ['a', 'b', 'c', 'd', 'e', 'f']
    assert result.opt_int == 5
    assert result.price == 10
    assert result.test_set == {'a', 'b'}
    assert len(result.matches_with_same_id) == 1
    assert len(result.matches_with_same_id[0].matches) == 2
    assert result.inner_doc.integer == 2
    assert result.inner_doc.l == ['c', 'd', 'a', 'b']
    assert result.test_dict == {'a': 10, 'b': 10, 'c': 3, 'd': 4}

    # doc1 is changed in place (no extra memory)
    assert doc1.text == 'hey here'
    assert doc1.categories == ['a', 'b', 'c', 'd', 'e', 'f']
    assert len(doc1.matches) == 2
    assert doc1.opt_int == 5
    assert doc1.price == 10
    assert doc1.test_set == {'a', 'b'}
    assert len(doc1.matches_with_same_id) == 1
    assert len(doc1.matches_with_same_id[0].matches) == 2
    assert doc1.inner_doc.integer == 2
    assert doc1.inner_doc.l == ['c', 'd', 'a', 'b']
    assert doc1.test_dict == {'a': 10, 'b': 10, 'c': 3, 'd': 4}


def test_reduce_different_ids():
    da1 = DocumentArray[MMDoc]([MMDoc() for _ in range(10)])
    da2 = DocumentArray[MMDoc]([MMDoc() for _ in range(10)])
    result = reduce(da1, da2)
    assert len(result) == 20
    # da1 is changed in place (no extra memory)
    assert len(da1) == 20


def test_reduce(doc1, doc2):
    da1 = DocumentArray[MMDoc]([doc1, MMDoc()])
    da2 = DocumentArray[MMDoc]([MMDoc(), doc2])
    result = reduce(da1, da2)
    assert len(result) == 3
    # da1 is changed in place (no extra memory)
    assert len(da1) == 3
    merged_doc = result[0]
    assert merged_doc.text == 'hey here'
    assert merged_doc.categories == ['a', 'b', 'c', 'd', 'e', 'f']
    assert len(merged_doc.matches) == 2
    assert merged_doc.opt_int == 5
    assert merged_doc.price == 10
    assert merged_doc.test_set == {'a', 'b'}
    assert len(merged_doc.matches_with_same_id) == 1
    assert len(merged_doc.matches_with_same_id[0].matches) == 2
    assert merged_doc.inner_doc.integer == 2
    assert merged_doc.inner_doc.l == ['c', 'd', 'a', 'b']


def test_reduce_all(doc1, doc2):
    da1 = DocumentArray[MMDoc]([doc1, MMDoc()])
    da2 = DocumentArray[MMDoc]([MMDoc(), doc2])
    da3 = DocumentArray[MMDoc]([MMDoc(), MMDoc(), doc1])
    result = reduce_all([da1, da2, da3])
    assert len(result) == 5
    # da1 is changed in place (no extra memory)
    assert len(da1) == 5
    merged_doc = result[0]
    assert merged_doc.text == 'hey here'
    assert merged_doc.categories == [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
    ]
    assert len(merged_doc.matches) == 2
    assert merged_doc.opt_int == 5
    assert merged_doc.price == 10
    assert merged_doc.test_set == {'a', 'b'}
    assert len(merged_doc.matches_with_same_id) == 1
    assert len(merged_doc.matches_with_same_id[0].matches) == 2
    assert merged_doc.inner_doc.integer == 2
    assert merged_doc.inner_doc.l == ['c', 'd', 'a', 'b', 'c', 'd', 'a', 'b']
