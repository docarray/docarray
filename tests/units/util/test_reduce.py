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
from typing import Dict, List, Optional, Set

import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.utils.reduce import reduce, reduce_all


class InnerDoc(BaseDoc):
    integer: int
    inner_list: List


class MMDoc(BaseDoc):
    text: str = ''
    price: int = 0
    categories: Optional[List[str]] = None
    image: Optional[ImageDoc] = None
    matches: Optional[DocList] = None
    matches_with_same_id: Optional[DocList] = None
    opt_int: Optional[int] = None
    test_set: Optional[Set] = None
    inner_doc: Optional[InnerDoc] = None
    test_dict: Optional[Dict] = None


@pytest.fixture
def doc1():
    return MMDoc(
        text='hey here',
        categories=['a', 'b', 'c'],
        price=10,
        matches=DocList[MMDoc]([MMDoc()]),
        matches_with_same_id=DocList[MMDoc](
            [MMDoc(id='a', matches=DocList[MMDoc]([MMDoc()]))]
        ),
        test_set={'a', 'a'},
        inner_doc=InnerDoc(integer=2, inner_list=['c', 'd']),
        test_dict={'a': 0, 'b': 2, 'd': 4, 'z': 3},
    )


@pytest.fixture
def doc2(doc1):
    return MMDoc(
        id=doc1.id,
        text='hey here 2',
        categories=['d', 'e', 'f'],
        price=5,
        opt_int=5,
        matches=DocList[MMDoc]([MMDoc()]),
        matches_with_same_id=DocList[MMDoc](
            [MMDoc(id='a', matches=DocList[MMDoc]([MMDoc()]))]
        ),
        test_set={'a', 'b'},
        inner_doc=InnerDoc(integer=3, inner_list=['a', 'b']),
        test_dict={'a': 10, 'b': 10, 'c': 3, 'z': None},
    )


def test_reduce_different_ids():
    da1 = DocList[MMDoc]([MMDoc() for _ in range(10)])
    da2 = DocList[MMDoc]([MMDoc() for _ in range(10)])
    result = reduce(da1, da2)
    assert len(result) == 20
    # da1 is changed in place (no extra memory)
    assert len(da1) == 20


def test_reduce(doc1, doc2):
    da1 = DocList[MMDoc]([doc1, MMDoc()])
    da2 = DocList[MMDoc]([MMDoc(), doc2])
    result = reduce(da1, da2)
    assert len(result) == 3
    # da1 is changed in place (no extra memory)
    assert len(da1) == 3
    merged_doc = result[0]
    assert merged_doc.text == 'hey here 2'
    assert merged_doc.categories == ['a', 'b', 'c', 'd', 'e', 'f']
    assert len(merged_doc.matches) == 2
    assert merged_doc.opt_int == 5
    assert merged_doc.price == 5
    assert merged_doc.test_set == {'a', 'b'}
    assert len(merged_doc.matches_with_same_id) == 1
    assert len(merged_doc.matches_with_same_id[0].matches) == 2
    assert merged_doc.inner_doc.integer == 3
    assert merged_doc.inner_doc.inner_list == ['c', 'd', 'a', 'b']


def test_reduce_all(doc1, doc2):
    da1 = DocList[MMDoc]([doc1, MMDoc()])
    da2 = DocList[MMDoc]([MMDoc(), doc2])
    da3 = DocList[MMDoc]([MMDoc(), MMDoc(), doc1])
    result = reduce_all([da1, da2, da3])
    assert len(result) == 5
    # da1 is changed in place (no extra memory)
    assert len(da1) == 5
    merged_doc = result[0]
    assert merged_doc.text == 'hey here 2'
    assert merged_doc.categories == [
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
    ]
    assert len(merged_doc.matches) == 2
    assert merged_doc.opt_int == 5
    assert merged_doc.price == 5
    assert merged_doc.test_set == {'a', 'b'}
    assert len(merged_doc.matches_with_same_id) == 1
    assert len(merged_doc.matches_with_same_id[0].matches) == 2
    assert merged_doc.inner_doc.integer == 3
    assert merged_doc.inner_doc.inner_list == ['c', 'd', 'a', 'b', 'c', 'd', 'a', 'b']


def test_update_ndarray():
    from docarray.typing import NdArray

    import numpy as np

    class MyDoc(BaseDoc):
        embedding: NdArray[128]

    embedding1 = np.random.rand(128)
    embedding2 = np.random.rand(128)

    doc1 = MyDoc(id='0', embedding=embedding1)
    doc2 = MyDoc(id='0', embedding=embedding2)
    doc1.update(doc2)
    assert (doc1.embedding == embedding2).all()
