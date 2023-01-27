from pydantic import parse_obj_as

from docarray import BaseDocument
from docarray.documents import Text


def test_simple_init():
    t = Text(text='hello')
    assert t.text == 'hello'


def test_str_init():
    t = parse_obj_as(Text, 'hello')
    assert t.text == 'hello'


def test_doc():
    class MyDoc(BaseDocument):
        text1: Text
        text2: Text

    doc = MyDoc(text1='hello', text2=Text(text='world'))

    assert doc.text1.text == 'hello'
    assert doc.text2.text == 'world'
