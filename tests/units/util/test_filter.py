import pytest
import json

from typing import Optional, List, Dict, Any
from docarray import BaseDocument, DocumentArray
from docarray.documents import Image, Text
from docarray.utils.filter import filter


class MMDoc(BaseDocument):
    text_doc: Text
    text: str = ''
    image: Optional[Image] = None
    price: int = 0
    optional_num: Optional[int] = None
    boolean: bool = False
    categories: Optional[List[str]] = None
    sub_docs: Optional[List[Text]] = None
    dictionary: Optional[Dict[str, Any]] = None


@pytest.fixture
def docs():
    mmdoc1 = MMDoc(
        text_doc=Text(text='Text Doc of Document 1'),
        text='Text of Document 1',
        sub_docs=[Text(text='subtext1'), Text(text='subtext2')],
        dictionary={},
    )
    mmdoc2 = MMDoc(
        text_doc=Text(text='Text Doc of Document 2'),
        text='Text of Document 2',
        image=Image(url='exampleimage.jpg'),
        price=3,
        dictionary={'a': 0, 'b': 1, 'c': 2},
    )
    mmdoc3 = MMDoc(
        text_doc=Text(text='Text Doc of Document 3'),
        text='Text of Document 3',
        price=1000,
        boolean=True,
        categories=['cat1', 'cat2'],
        sub_docs=[Text(text='subtext1'), Text(text='subtext2')],
        optional_num=30,
        dictionary={'a': 0, 'b': 1},
    )
    docs = DocumentArray[MMDoc]([mmdoc1, mmdoc2, mmdoc3])

    return docs


@pytest.mark.parametrize('dict_api', [True, False])
def test_empty_filter(docs, dict_api):
    q = {} if dict_api else '{}'
    result = filter(docs, q)
    assert len(result) == len(docs)


@pytest.mark.parametrize('dict_api', [True, False])
def test_simple_filter(docs, dict_api):
    if dict_api:
        method = lambda query: filter(docs, query)
    else:
        method = lambda query: filter(docs, json.dumps(query))

    result = method({'text': {'$eq': 'Text of Document 1'}})
    assert len(result) == 1
    assert result[0].text == 'Text of Document 1'

    result = method({'text': {'$neq': 'Text of Document 1'}})
    assert len(result) == 2

    result = method({'text_doc': {'$eq': 'Text Doc of Document 1'}})
    assert len(result) == 1
    assert result[0].text_doc == 'Text Doc of Document 1'

    result = method({'text_doc': {'$neq': 'Text Doc of Document 1'}})
    assert len(result) == 2

    result = method({'text': {'$regex': 'Text*'}})
    assert len(result) == 3

    result = method({'text': {'$regex': 'TeAxt*'}})
    assert len(result) == 0

    result = method({'text_doc': {'$regex': 'Text*'}})
    assert len(result) == 3

    result = method({'text_doc': {'$regex': 'TeAxt*'}})
    assert len(result) == 0

    result = method({'price': {'$gte': 500}})
    assert len(result) == 1

    result = method({'price': {'$lte': 500}})
    assert len(result) == 2

    result = method({'dictionary': {'$eq': {}}})
    assert len(result) == 1
    assert result[0].dictionary == {}

    result = method({'dictionary': {'$eq': {'a': 0, 'b': 1}}})
    assert len(result) == 1
    assert result[0].dictionary == {'a': 0, 'b': 1}

    result = method({'text': {'$neq': 'Text of Document 1'}})
    assert len(result) == 2

    # EXISTS DOES NOT SEEM TO WORK
    result = method({'optional_num': {'$exists': True}})
    assert len(result) == 3
    result = method({'optional_num': {'$exists': False}})
    assert len(result) == 0

    result = method({'price': {'$exists': True}})
    assert len(result) == 3
    result = method({'price': {'$exists': False}})
    assert len(result) == 0

    # DOES NOT SEEM TO WORK WITH OPTIONAL NUMBERS
    result = method({'optional_num': {'$gte': 20}})
    assert len(result) == 1

    result = method({'optional_num': {'$lte': 20}})
    assert len(result) == 0


@pytest.mark.parametrize('dict_api', [True, False])
def test_nested_filter(docs, dict_api):
    if dict_api:
        method = lambda query: filter(docs, query)
    else:
        method = lambda query: filter(docs, json.dumps(query))

    result = method({'dictionary.a': {'$eq': 0}})
    assert len(result) == 2
    for res in result:
        assert res.dictionary['a'] == 0

    result = method({'dictionary.c': {'$exists': True}})
    assert len(result) == 1
    assert result[0].dictionary['c'] == 2

    result = method({'image.url': {'$eq': 'exampleimage.jpg'}})
    assert len(result) == 1
    assert result[0].image.url == 'exampleimage.jpg'


@pytest.mark.parametrize('dict_api', [True, False])
def test_array_simple_filters(docs, dict_api):
    if dict_api:
        method = lambda query: filter(docs, query)
    else:
        method = lambda query: filter(docs, json.dumps(query))

    # SIZE DOES NOT SEEM TO WORK
    result = method({'sub_docs': {'$size': 2}})
    assert len(result) == 2

    result = method({'categories': {'$size': 2}})
    assert len(result) == 1


@pytest.mark.parametrize('dict_api', [True, False])
def test_placehold_filter(dict_api):
    docs = DocumentArray[MMDoc](
        [
            MMDoc(text='A', text_doc=Text(text='A')),
            MMDoc(text='A', text_doc=Text(text='B')),
        ]
    )

    if dict_api:
        method = lambda query: filter(docs, query)
    else:
        method = lambda query: filter(docs, json.dumps(query))

    # DOES NOT SEEM TO WORK
    result = method({'text': {'$eq': '{text_doc}'}})
    assert len(result) == 1

    result = method({'text_doc': {'$eq': '{text}'}})
    assert len(result) == 1


@pytest.mark.parametrize('dict_api', [True, False])
def test_logic_filter(docs, dict_api):
    if dict_api:
        method = lambda query: filter(docs, query)
    else:
        method = lambda query: filter(docs, json.dumps(query))
    result = method(
        {
            '$or': {
                'text': {'$eq': 'Text of Document 1'},
                'text_doc': {'$eq': 'Text Doc of Document 2'},
            }
        }
    )
    assert len(result) == 2

    result = method(
        {
            '$not': {
                '$or': {
                    'text': {'$eq': 'Text of Document 1'},
                    'text_doc': {'$eq': 'Text Doc of Document 2'},
                }
            }
        }
    )
    assert len(result) == 1

    result = method(
        {
            '$and': {
                'text': {'$eq': 'Text of Document 1'},
                'text_doc': {'$eq': 'Text Doc of Document 2'},
            }
        }
    )
    assert len(result) == 0

    result = method(
        {
            '$not': {
                '$and': {
                    'text': {'$eq': 'Text of Document 1'},
                    'text_doc': {'$eq': 'Text Doc of Document 2'},
                }
            }
        }
    )
    assert len(result) == 3
