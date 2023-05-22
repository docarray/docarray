from typing import Optional

import pandas as pd
import pytest

from docarray import BaseDoc, DocList, DocVec
from docarray.documents import ImageDoc


@pytest.fixture()
def nested_doc_cls():
    class MyDoc(BaseDoc):
        count: Optional[int]
        text: str

    class MyDocNested(MyDoc):
        image: ImageDoc

    return MyDocNested


@pytest.mark.parametrize('doc_vec', [False, True])
def test_to_from_pandas_df(nested_doc_cls, doc_vec):
    da = DocList[nested_doc_cls](
        [
            nested_doc_cls(
                count=0,
                text='hello',
                image=ImageDoc(url='aux.png'),
            ),
            nested_doc_cls(text='hello world', image=ImageDoc()),
        ]
    )
    if doc_vec:
        da = da.to_doc_vec()
    df = da.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert (
        df.columns
        == [
            'id',
            'count',
            'text',
            'image__id',
            'image__url',
            'image__tensor',
            'image__embedding',
            'image__bytes_',
        ]
    ).all()

    if doc_vec:
        da_from_df = DocVec[nested_doc_cls].from_dataframe(df)
        assert isinstance(da_from_df, DocVec)
    else:
        da_from_df = DocList[nested_doc_cls].from_dataframe(df)
        assert isinstance(da_from_df, DocList)
    for doc1, doc2 in zip(da, da_from_df):
        assert doc1 == doc2


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


@pytest.mark.parametrize('array_cls', [DocList, DocVec])
def test_from_pandas_without_schema_raise_exception(array_cls):
    with pytest.raises(TypeError, match='no document schema defined'):
        df = pd.DataFrame(
            columns=['title', 'count'], data=[['title 0', 0], ['title 1', 1]]
        )
        array_cls.from_dataframe(df=df)


@pytest.mark.parametrize('array_cls', [DocList, DocVec])
def test_from_pandas_with_wrong_schema_raise_exception(nested_doc, array_cls):
    with pytest.raises(ValueError, match='Column names do not match the schema'):
        df = pd.DataFrame(
            columns=['title', 'count'], data=[['title 0', 0], ['title 1', 1]]
        )
        array_cls[nested_doc.__class__].from_dataframe(df=df)


def test_doc_list_error():
    class Book(BaseDoc):
        title: str

    # not testing DocVec bc it already fails here (as it should!)
    docs = DocList([Book(title='hello'), Book(title='world')])
    with pytest.raises(TypeError):
        docs.to_dataframe()
