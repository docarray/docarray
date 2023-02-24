from typing import Optional

import pandas as pd
import pytest

from docarray import BaseDocument, DocumentArray
from docarray.documents import Image


@pytest.fixture()
def nested_doc_cls():
    class MyDoc(BaseDocument):
        count: Optional[int]
        text: str

    class MyDocNested(MyDoc):
        image: Image

    return MyDocNested


def test_to_from_pandas_df(nested_doc_cls):
    da = DocumentArray[nested_doc_cls](
        [
            nested_doc_cls(
                count=0,
                text='hello',
                image=Image(url='aux.png'),
            ),
            nested_doc_cls(text='hello world', image=Image()),
        ]
    )
    df = da.to_pandas()
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
            'image__bytes',
        ]
    ).all()

    da_from_df = DocumentArray[nested_doc_cls].from_pandas(df)
    for doc1, doc2 in zip(da, da_from_df):
        assert doc1 == doc2


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


def test_from_pandas_without_schema_raise_exception():
    with pytest.raises(TypeError, match='no document schema defined'):
        df = pd.DataFrame(
            columns=['title', 'count'], data=[['title 0', 0], ['title 1', 1]]
        )
        DocumentArray.from_pandas(df=df)


def test_from_pandas_with_wrong_schema_raise_exception(nested_doc):
    with pytest.raises(ValueError, match='Column names do not match the schema'):
        df = pd.DataFrame(
            columns=['title', 'count'], data=[['title 0', 0], ['title 1', 1]]
        )
        DocumentArray[nested_doc.__class__].from_pandas(df=df)
