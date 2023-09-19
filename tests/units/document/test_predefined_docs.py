from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import TextDoc


def test_nested_defaut_value():
    class MyDocument(BaseDoc):
        caption: TextDoc

    doc = MyDocument(caption='A tiger in the jungle')

    assert doc.caption.text == 'A tiger in the jungle'


def test_parse_default_val():

    doc = parse_obj_as(TextDoc, 'A tiger in the jungle')
    assert doc.text == 'A tiger in the jungle'


def test_parse_default_val_init():

    doc = TextDoc('A tiger in the jungle')
    assert doc.text == 'A tiger in the jungle'
