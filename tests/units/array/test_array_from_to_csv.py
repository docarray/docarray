import os
from typing import Optional

import pytest

from docarray import BaseDocument, DocumentArray
from docarray.array.array.io import _assert_schema
from docarray.documents import Image
from tests import TOYDATA_DIR


class MyDoc(BaseDocument):
    count: int
    text: str


class MyDocNested(MyDoc):
    image: Image
    image2: Image


def test_to_csv(tmpdir):
    da = DocumentArray[MyDocNested](
        [
            MyDocNested(
                text='hello', image=Image(url='aux.png'), image2=Image(url='aux.png')
            ),
            MyDocNested(text='hello world', image=Image(), image2=Image()),
        ]
    )
    tmp_file = str(tmpdir / 'tmp.csv')
    da.to_csv(tmp_file)
    assert os.path.isfile(tmp_file)


def test_from_csv_nested():
    da = DocumentArray[MyDocNested].from_csv(
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
    assert da[1].image2.url == 'image_11.png'
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
    with pytest.raises(ValueError, match=r'.*Outer.*embedding.*text.*image.*'):
        DocumentArray[nested_doc.__class__].from_csv(
            file_path=str(TOYDATA_DIR / 'docs_nested.csv')
        )


def test_assert_schema(nested_doc):
    assert _assert_schema(nested_doc.__class__, 'img')
    assert _assert_schema(nested_doc.__class__, 'middle.img')
    assert _assert_schema(nested_doc.__class__, 'middle.inner.img')
    assert _assert_schema(nested_doc.__class__, 'middle')
    assert not _assert_schema(nested_doc.__class__, 'inner')
