from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import TextDoc


def test_simple_init():
    t = TextDoc(text='hello')
    assert t.text == 'hello'


def test_str_init():
    t = parse_obj_as(TextDoc, 'hello')
    assert t.text == 'hello'


def test_doc():
    class MyDoc(BaseDoc):
        text1: TextDoc
        text2: TextDoc

    doc = MyDoc(text1='hello', text2=TextDoc(text='world'))

    assert doc.text1.text == 'hello'
    assert doc.text2.text == 'world'


def test_doc_2():
    class MyDoc(BaseDoc):
        text1: TextDoc
        text2: TextDoc

    doc = MyDoc(text1='hello1', text2='hello2')

    assert doc.text1.text == 'hello1'
    assert doc.text2.text == 'hello2'
