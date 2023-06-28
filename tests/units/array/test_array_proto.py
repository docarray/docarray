import numpy as np
import pytest

from docarray import BaseDoc, DocList
from docarray.base_doc import AnyDoc
from docarray.documents import ImageDoc, TextDoc
from docarray.typing import NdArray


@pytest.mark.proto
def test_simple_proto():
    class CustomDoc(BaseDoc):
        text: str
        tensor: NdArray

    da = DocList(
        [CustomDoc(text='hello', tensor=np.zeros((3, 224, 224))) for _ in range(10)]
    )

    new_da = DocList[CustomDoc].from_protobuf(da.to_protobuf())

    for doc1, doc2 in zip(da, new_da):
        assert doc1.text == doc2.text
        assert (doc1.tensor == doc2.tensor).all()


@pytest.mark.proto
def test_nested_proto():
    class CustomDocument(BaseDoc):
        text: TextDoc
        image: ImageDoc

    da = DocList[CustomDocument](
        [
            CustomDocument(
                text=TextDoc(text='hello'),
                image=ImageDoc(tensor=np.zeros((3, 224, 224))),
            )
            for _ in range(10)
        ]
    )

    DocList[CustomDocument].from_protobuf(da.to_protobuf())


@pytest.mark.proto
def test_nested_proto_any_doc():
    class CustomDocument(BaseDoc):
        text: TextDoc
        image: ImageDoc

    da = DocList[CustomDocument](
        [
            CustomDocument(
                text=TextDoc(text='hello'),
                image=ImageDoc(tensor=np.zeros((3, 224, 224))),
            )
            for _ in range(10)
        ]
    )

    DocList.from_protobuf(da.to_protobuf())


@pytest.mark.proto
def test_any_doc_list_proto():
    doc = AnyDoc(hello='world')
    pt = DocList([doc]).to_protobuf()
    docs = DocList.from_protobuf(pt)
    assert docs[0].dict()['hello'] == 'world'


@pytest.mark.proto
def test_any_nested_doc_list_proto():
    from docarray import BaseDoc, DocList

    class TextDocWithId(BaseDoc):
        id: str
        text: str

    class ResultTestDoc(BaseDoc):
        matches: DocList[TextDocWithId]

    index_da = DocList[TextDocWithId](
        [TextDocWithId(id=f'{i}', text=f'ID {i}') for i in range(10)]
    )

    out_da = DocList[ResultTestDoc]([ResultTestDoc(matches=index_da[0:2])])
    pb = out_da.to_protobuf()
    docs = DocList.from_protobuf(pb)
    assert docs[0].matches[0].id == '0'
    assert len(docs[0].matches) == 2
    assert len(docs) == 1


@pytest.mark.proto
def test_union_type_error():
    from typing import Union

    class CustomDoc(BaseDoc):
        ud: Union[TextDoc, ImageDoc] = TextDoc(text='union type')

    docs = DocList[CustomDoc]([CustomDoc(ud=TextDoc(text='union type'))])

    with pytest.raises(ValueError):
        DocList[CustomDoc].from_protobuf(docs.to_protobuf())

    class BasisUnion(BaseDoc):
        ud: Union[int, str]

    docs_basic = DocList[BasisUnion]([BasisUnion(ud="hello")])
    docs_copy = DocList[BasisUnion].from_protobuf(docs_basic.to_protobuf())
    assert docs_copy == docs_basic
