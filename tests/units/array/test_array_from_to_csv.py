import os
from typing import Optional

import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from tests import TOYDATA_DIR


@pytest.fixture()
def nested_doc_cls():
    class MyDoc(BaseDoc):
        count: Optional[int]
        text: str

    class MyDocNested(MyDoc):
        image: ImageDoc
        image2: ImageDoc

    return MyDocNested


def test_to_from_csv(tmpdir, nested_doc_cls):
    da = DocList[nested_doc_cls](
        [
            nested_doc_cls(
                count=0,
                text='hello',
                image=ImageDoc(url='aux.png'),
                image2=ImageDoc(url='aux.png'),
            ),
            nested_doc_cls(text='hello world', image=ImageDoc(), image2=ImageDoc()),
        ]
    )
    tmp_file = str(tmpdir / 'tmp.csv')
    da.to_csv(tmp_file)
    assert os.path.isfile(tmp_file)

    da_from = DocList[nested_doc_cls].from_csv(tmp_file)
    for doc1, doc2 in zip(da, da_from):
        assert doc1 == doc2


def test_from_csv_nested(nested_doc_cls):
    da = DocList[nested_doc_cls].from_csv(
        file_path=str(TOYDATA_DIR / 'docs_nested.csv')
    )
    assert len(da) == 3

    for i, doc in enumerate(da):
        assert doc.count.__class__ == int
        assert doc.count == int(f'{i}{i}{i}')

        assert doc.text.__class__ == str
        assert doc.text == f'hello {i}'

        assert doc.image.__class__ == ImageDoc
        assert doc.image.tensor is None
        assert doc.image.embedding is None
        assert doc.image.bytes_ is None

        assert doc.image2.__class__ == ImageDoc
        assert doc.image2.tensor is None
        assert doc.image2.embedding is None
        assert doc.image2.bytes_ is None

    assert da[0].image2.url == 'image_10.png'
    assert da[1].image2.url is None
    assert da[2].image2.url is None


@pytest.fixture()
def nested_doc():
    class Inner(BaseDoc):
        img: Optional[ImageDoc]

    class Middle(BaseDoc):
        img: Optional[ImageDoc]
        inner: Optional[Inner]

    class Outer(BaseDoc):
        img: Optional[ImageDoc]
        middle: Optional[Middle]

    doc = Outer(
        img=ImageDoc(), middle=Middle(img=ImageDoc(), inner=Inner(img=ImageDoc()))
    )
    return doc


def test_from_csv_without_schema_raise_exception():
    with pytest.raises(TypeError, match='no document schema defined'):
        DocList.from_csv(file_path=str(TOYDATA_DIR / 'docs_nested.csv'))


def test_from_csv_with_wrong_schema_raise_exception(nested_doc):
    with pytest.raises(ValueError, match='Column names do not match the schema'):
        DocList[nested_doc.__class__].from_csv(file_path=str(TOYDATA_DIR / 'docs.csv'))


def test_from_remote_csv_file():
    remote_url = 'https://github.com/docarray/docarray/blob/main/tests/toydata/books.csv?raw=true'

    class Book(BaseDoc):
        title: str
        author: str
        year: int

    books = DocList[Book].from_csv(file_path=remote_url)

    assert len(books) == 3


def test_doc_list_error(tmpdir):
    class Book(BaseDoc):
        title: str

    docs = DocList([Book(title='hello'), Book(title='world')])
    tmp_file = str(tmpdir / 'tmp.csv')
    with pytest.raises(TypeError):
        docs.to_csv(tmp_file)


def test_union_type_error(tmp_path):
    from typing import Union

    from docarray.documents import TextDoc

    class CustomDoc(BaseDoc):
        ud: Union[TextDoc, ImageDoc] = TextDoc(text='union type')

    docs = DocList[CustomDoc]([CustomDoc(ud=TextDoc(text='union type'))])

    with pytest.raises(ValueError):
        docs.to_csv(str(tmp_path) + ".csv")
        DocList[CustomDoc].from_csv(str(tmp_path) + ".csv")

    class BasisUnion(BaseDoc):
        ud: Union[int, str]

    docs_basic = DocList[BasisUnion]([BasisUnion(ud="hello")])
    docs_basic.to_csv(str(tmp_path) + ".csv")
    docs_copy = DocList[BasisUnion].from_csv(str(tmp_path) + ".csv")
    assert docs_copy == docs_basic
