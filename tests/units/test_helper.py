from typing import Optional

import pytest

from docarray import BaseDocument
from docarray.documents import Image
from docarray.helper import (
    _access_path_to_dict,
    _dict_to_access_paths,
    _update_nested_dicts,
    is_access_path_valid,
)


@pytest.fixture()
def nested_doc():
    class Inner(BaseDocument):
        img: Optional[Image]

    class Middle(BaseDocument):
        img: Optional[Image]
        inner: Optional[Inner]

    class Outer(BaseDocument):
        img: Optional[Image]
        middle: Optional[Middle]

    doc = Outer(img=Image(), middle=Middle(img=Image(), inner=Inner(img=Image())))
    return doc


def test_is_access_path_valid(nested_doc):
    assert is_access_path_valid(nested_doc.__class__, 'img')
    assert is_access_path_valid(nested_doc.__class__, 'middle__img')
    assert is_access_path_valid(nested_doc.__class__, 'middle__inner__img')
    assert is_access_path_valid(nested_doc.__class__, 'middle')


def test_is_access_path_not_valid(nested_doc):
    assert not is_access_path_valid(nested_doc.__class__, 'inner')
    assert not is_access_path_valid(nested_doc.__class__, 'some__other__path')
    assert not is_access_path_valid(nested_doc.__class__, 'middle.inner')


def test_get_access_paths():
    class Painting(BaseDocument):
        title: str
        img: Image

    access_paths = Painting._get_access_paths()
    assert access_paths == [
        'id',
        'title',
        'img__id',
        'img__url',
        'img__tensor',
        'img__embedding',
        'img__bytes',
    ]


def test_dict_to_access_paths():
    d = {
        'a0': {'b0': {'c0': 0}, 'b1': {'c0': 1}},
        'a1': {'b0': {'c0': 2, 'c1': 3}, 'b1': 4},
    }
    casted = _dict_to_access_paths(d)
    assert casted == {
        'a0__b0__c0': 0,
        'a0__b1__c0': 1,
        'a1__b0__c0': 2,
        'a1__b0__c1': 3,
        'a1__b1': 4,
    }


def test_access_path_to_dict():
    access_path = 'a__b__c__d__e'
    value = 1
    result = {'a': {'b': {'c': {'d': {'e': value}}}}}
    assert _access_path_to_dict(access_path, value) == result


def test_update_nested_dict():
    d1 = {'text': 'hello', 'image': {'tensor': None}}
    d2 = {'image': {'url': 'some.png'}}

    _update_nested_dicts(d1, d2)
    assert d1 == {'text': 'hello', 'image': {'tensor': None, 'url': 'some.png'}}
