from typing import Optional

import pytest

from docarray import BaseDocument
from docarray.documents import ImageDoc
from docarray.helper import (
    _access_path_dict_to_nested_dict,
    _access_path_to_dict,
    _dict_to_access_paths,
    _is_access_path_valid,
    _update_nested_dicts,
)


@pytest.fixture()
def nested_doc():
    class Inner(BaseDocument):
        img: Optional[ImageDoc]

    class Middle(BaseDocument):
        img: Optional[ImageDoc]
        inner: Optional[Inner]

    class Outer(BaseDocument):
        img: Optional[ImageDoc]
        middle: Optional[Middle]
        da: DocumentArray[Inner]

    doc = Outer(
        img=ImageDoc(), middle=Middle(img=ImageDoc(), inner=Inner(img=ImageDoc()))
    )
    return doc


def test_is_access_path_valid(nested_doc):
    assert _is_access_path_valid(nested_doc.__class__, 'img')
    assert _is_access_path_valid(nested_doc.__class__, 'middle__img')
    assert _is_access_path_valid(nested_doc.__class__, 'middle__inner__img')
    assert _is_access_path_valid(nested_doc.__class__, 'middle')
    assert _is_access_path_valid(nested_doc.__class__, 'da__img__url')


def test_is_access_path_not_valid(nested_doc):
    assert not _is_access_path_valid(nested_doc.__class__, 'inner')
    assert not _is_access_path_valid(nested_doc.__class__, 'some__other__path')
    assert not _is_access_path_valid(nested_doc.__class__, 'middle.inner')


def test_get_access_paths():
    class Painting(BaseDocument):
        title: str
        img: ImageDoc

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


def test_access_path_dict_to_nested_dict():
    d = {
        'a0__b0__c0': 0,
        'a0__b1__c0': 1,
        'a1__b0__c0': 2,
        'a1__b0__c1': 3,
        'a1__b1': 4,
    }
    casted = _access_path_dict_to_nested_dict(d)
    assert casted == {
        'a0': {'b0': {'c0': 0}, 'b1': {'c0': 1}},
        'a1': {'b0': {'c0': 2, 'c1': 3}, 'b1': 4},
    }


def test_update_nested_dict():
    d1 = {'text': 'hello', 'image': {'tensor': None}}
    d2 = {'image': {'url': 'some.png'}}

    _update_nested_dicts(d1, d2)
    assert d1 == {'text': 'hello', 'image': {'tensor': None, 'url': 'some.png'}}
