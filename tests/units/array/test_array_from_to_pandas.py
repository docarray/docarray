from typing import List, Optional

import pandas as pd
import pytest

from docarray import BaseDoc, DocList, DocVec
from docarray.documents import ImageDoc
from docarray.typing import NdArray, TorchTensor
from docarray.utils._internal.pydantic import is_pydantic_v2


@pytest.fixture()
def nested_doc_cls():
    class MyDoc(BaseDoc):
        count: Optional[int]
        text: str

    class MyDocNested(MyDoc):
        image: ImageDoc
        lst: List[str]

    return MyDocNested


@pytest.mark.skipif(is_pydantic_v2, reason="Not working with pydantic v2")
@pytest.mark.parametrize('doc_vec', [False, True])
def test_to_from_pandas_df(nested_doc_cls, doc_vec):
    da = DocList[nested_doc_cls](
        [
            nested_doc_cls(
                count=0,
                text='hello',
                image=ImageDoc(url='aux.png'),
                lst=["hello", "world"],
            ),
            nested_doc_cls(
                text='hello world', image=ImageDoc(), lst=["hello", "world"]
            ),
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
            'lst',
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


@pytest.mark.proto
def test_union_type_error():
    from typing import Union

    from docarray.documents import TextDoc

    class CustomDoc(BaseDoc):
        ud: Union[TextDoc, ImageDoc] = TextDoc(text='union type')

    docs = DocList[CustomDoc]([CustomDoc(ud=TextDoc(text='union type'))])

    with pytest.raises(ValueError):
        DocList[CustomDoc].from_dataframe(docs.to_dataframe())

    class BasisUnion(BaseDoc):
        ud: Union[int, str]

    docs_basic = DocList[BasisUnion]([BasisUnion(ud="hello")])
    docs_copy = DocList[BasisUnion].from_dataframe(docs_basic.to_dataframe())
    assert docs_copy == docs_basic


@pytest.mark.parametrize('tensor_type', [NdArray, TorchTensor])
def test_from_to_pandas_tensor_type(tensor_type):
    class MyDoc(BaseDoc):
        embedding: tensor_type
        text: str
        image: ImageDoc

    da = DocVec[MyDoc](
        [
            MyDoc(
                embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png')
            ),
            MyDoc(embedding=[5, 4, 3, 2, 1], text='hello world', image=ImageDoc()),
        ],
        tensor_type=tensor_type,
    )
    df_da = da.to_dataframe()
    da2 = DocVec[MyDoc].from_dataframe(df_da, tensor_type=tensor_type)
    assert da2.tensor_type == tensor_type
    assert isinstance(da2.embedding, tensor_type)
