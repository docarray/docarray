import os
from typing import Optional

import pytest

from docarray import BaseDocument, DocumentArray
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
