// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
                                                 // "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import numpy as np
import pytest
from typing import Dict, List

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
    assert docs[0].hello == 'world'


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


class MySimpleDoc(BaseDoc):
    title: str


class MyComplexDoc(BaseDoc):
    content_dict_doclist: Dict[str, DocList[MySimpleDoc]]
    content_dict_list: Dict[str, List[MySimpleDoc]]
    aux_dict: Dict[str, int]


def test_to_from_proto_complex():
    da = DocList[MyComplexDoc](
        [
            MyComplexDoc(
                content_dict_doclist={
                    'test1': DocList[MySimpleDoc](
                        [MySimpleDoc(title='123'), MySimpleDoc(title='456')]
                    )
                },
                content_dict_list={
                    'test1': [MySimpleDoc(title='123'), MySimpleDoc(title='456')]
                },
                aux_dict={'a': 0},
            )
        ]
    )
    da2 = DocList[MyComplexDoc].from_protobuf(da.to_protobuf())
    assert len(da2) == 1
    d2 = da2[0]
    assert d2.aux_dict == {'a': 0}
    assert len(d2.content_dict_doclist['test1']) == 2
    assert d2.content_dict_doclist['test1'][0].title == '123'
    assert d2.content_dict_doclist['test1'][1].title == '456'
    assert len(d2.content_dict_list['test1']) == 2
    assert d2.content_dict_list['test1'][0].title == '123'
    assert d2.content_dict_list['test1'][1].title == '456'
