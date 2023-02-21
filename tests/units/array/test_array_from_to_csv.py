import os
from typing import Optional

import pytest

from docarray import BaseDocument, DocumentArray
from docarray.array.array.io import (
    _dict_to_access_paths,
    _update_nested_dicts,
    is_access_path_valid,
)
from docarray.documents import Image
from tests import TOYDATA_DIR


@pytest.fixture()
def nested_doc_cls():
    class MyDoc(BaseDocument):
        count: Optional[int]
        text: str

    class MyDocNested(MyDoc):
        image: Image
        image2: Image

    return MyDocNested


def test_to_from_csv(tmpdir, nested_doc_cls):
    da = DocumentArray[nested_doc_cls](
        [
            nested_doc_cls(
                count=0,
                text='hello',
                image=Image(url='aux.png'),
                image2=Image(url='aux.png'),
            ),
            nested_doc_cls(text='hello world', image=Image(), image2=Image()),
        ]
    )
    tmp_file = str(tmpdir / 'tmp.csv')
    da.to_csv(tmp_file)
    assert os.path.isfile(tmp_file)

    da_from = DocumentArray[nested_doc_cls].from_csv(tmp_file)
    for doc1, doc2 in zip(da, da_from):
        assert doc1 == doc2


def test_from_csv_nested(nested_doc_cls):
    da = DocumentArray[nested_doc_cls].from_csv(
        file_path=str(TOYDATA_DIR / 'docs_nested.csv')
    )
    assert len(da) == 3

    for i, doc in enumerate(da):
        assert doc.count.__class__ == int
        assert doc.count == int(f'{i}{i}{i}')

        assert doc.text.__class__ == str
        assert doc.text == f'hello {i}'

        assert doc.image.__class__ == Image
        assert doc.image.tensor is None
        assert doc.image.embedding is None
        assert doc.image.bytes is None

        assert doc.image2.__class__ == Image
        assert doc.image2.tensor is None
        assert doc.image2.embedding is None
        assert doc.image2.bytes is None

    assert da[0].image2.url == 'image_10.png'
    assert da[1].image2.url is None
    assert da[2].image2.url is None


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


def test_from_csv_without_schema_raise_exception():
    with pytest.raises(TypeError, match='no document schema defined'):
        DocumentArray.from_csv(file_path=str(TOYDATA_DIR / 'docs_nested.csv'))


def test_from_csv_with_wrong_schema_raise_exception(nested_doc):
    with pytest.raises(
        ValueError, match='Fields provided in the csv file do not match the schema'
    ):
        DocumentArray[nested_doc.__class__].from_csv(
            file_path=str(TOYDATA_DIR / 'docs.csv')
        )


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


def test_is_access_path_valid(nested_doc):
    assert is_access_path_valid(nested_doc.__class__, 'img')
    assert is_access_path_valid(nested_doc.__class__, 'middle__img')
    assert is_access_path_valid(nested_doc.__class__, 'middle__inner__img')
    assert is_access_path_valid(nested_doc.__class__, 'middle')
    assert not is_access_path_valid(nested_doc.__class__, 'inner')
    assert not is_access_path_valid(nested_doc.__class__, 'some__other__path')
    assert not is_access_path_valid(nested_doc.__class__, 'middle.inner')


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


def test_update_nested_dict():
    d1 = {'text': 'hello', 'image': {'tensor': None}}
    d2 = {'image': {'url': 'some.png'}}

    _update_nested_dicts(d1, d2)
    assert d1 == {'text': 'hello', 'image': {'tensor': None, 'url': 'some.png'}}
