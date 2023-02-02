from typing import Optional, List, Dict, Any, Set
from docarray import BaseDocument, DocumentArray
from docarray.documents import Image
from docarray.utils.reduce import reduce_docs


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
    dictionary: Optional[Dict[str, Any]] = None
    opt_int: Optional[int] = None
    test_set: Optional[Set] = None
    inner_doc: Optional[InnerDoc] = None


def test_simple_reduce_arrays_concatenated():
    doc1 = MMDoc(
        text='hey here',
        categories=['a', 'b', 'c'],
        price=10,
        matches=DocumentArray[MMDoc]([MMDoc()]),
        matches_with_same_id=DocumentArray[MMDoc]([MMDoc(id='a', matches=DocumentArray[MMDoc]([MMDoc()]))]),
        test_set={'a', 'a'},
        inner_doc=InnerDoc(integer=2, l=['c', 'd']))
    doc2 = MMDoc(
        id=doc1.id,
        text='hey here 2',
        categories=['d', 'e', 'f'],
        price=5,
        opt_int=5,
        matches=DocumentArray[MMDoc]([MMDoc()]),
        matches_with_same_id=DocumentArray[MMDoc]([MMDoc(id='a', matches=DocumentArray[MMDoc]([MMDoc()]))]),
        test_set={'a', 'b'},
        inner_doc=InnerDoc(integer=3, l=['a', 'b']))

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

