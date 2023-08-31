import pytest
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import TextDoc
from docarray.utils._internal.pydantic import is_pydantic_v2


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_simple_init():
    t = TextDoc(text='hello')
    assert t.text == 'hello'


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_str_init():
    t = parse_obj_as(TextDoc, 'hello')
    assert t.text == 'hello'


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2 for now")
def test_doc():
    class MyDoc(BaseDoc):
        text1: TextDoc
        text2: TextDoc

    doc = MyDoc(text1='hello', text2=TextDoc(text='world'))

    assert doc.text1.text == 'hello'
    assert doc.text2.text == 'world'
