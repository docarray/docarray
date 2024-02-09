# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from typing import Dict, List

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.typing import NdArray


class MyDoc(BaseDoc):
    embedding: NdArray
    text: str
    image: ImageDoc


class MySimpleDoc(BaseDoc):
    title: str


class MyComplexDoc(BaseDoc):
    content_dict_doclist: Dict[str, DocList[MySimpleDoc]]
    content_dict_list: Dict[str, List[MySimpleDoc]]
    aux_dict: Dict[str, int]


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes(protocol, compress):
    d = MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png'))

    assert d.text == 'hello'
    assert d.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d.image.url == 'aux.png'
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    d2 = MyDoc.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.text == 'hello'
    assert d2.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d2.image.url == 'aux.png'


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_base64(protocol, compress):
    d = MyDoc(embedding=[1, 2, 3, 4, 5], text='hello', image=ImageDoc(url='aux.png'))

    assert d.text == 'hello'
    assert d.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d.image.url == 'aux.png'
    bstr = d.to_base64(protocol=protocol, compress=compress)
    d2 = MyDoc.from_base64(bstr, protocol=protocol, compress=compress)
    assert d2.text == 'hello'
    assert d2.embedding.tolist() == [1, 2, 3, 4, 5]
    assert d2.image.url == 'aux.png'


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_bytes_complex(protocol, compress):
    d = MyComplexDoc(
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
    bstr = d.to_bytes(protocol=protocol, compress=compress)
    d2 = MyComplexDoc.from_bytes(bstr, protocol=protocol, compress=compress)
    assert d2.aux_dict == {'a': 0}
    assert len(d2.content_dict_doclist['test1']) == 2
    assert d2.content_dict_doclist['test1'][0].title == '123'
    assert d2.content_dict_doclist['test1'][1].title == '456'
    assert len(d2.content_dict_list['test1']) == 2
    assert d2.content_dict_list['test1'][0].title == '123'
    assert d2.content_dict_list['test1'][1].title == '456'


@pytest.mark.parametrize('protocol', ['protobuf', 'pickle'])
@pytest.mark.parametrize('compress', ['lz4', 'bz2', 'lzma', 'zlib', 'gzip', None])
def test_to_from_base64_complex(protocol, compress):
    d = MyComplexDoc(
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
    bstr = d.to_base64(protocol=protocol, compress=compress)
    d2 = MyComplexDoc.from_base64(bstr, protocol=protocol, compress=compress)
    assert d2.aux_dict == {'a': 0}
    assert len(d2.content_dict_doclist['test1']) == 2
    assert d2.content_dict_doclist['test1'][0].title == '123'
    assert d2.content_dict_doclist['test1'][1].title == '456'
    assert len(d2.content_dict_list['test1']) == 2
    assert d2.content_dict_list['test1'][0].title == '123'
    assert d2.content_dict_list['test1'][1].title == '456'
