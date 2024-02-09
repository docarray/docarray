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
from typing import Optional

import pytest

from docarray import BaseDoc, DocList
from docarray.documents import ImageDoc
from docarray.helper import (
    _access_path_dict_to_nested_dict,
    _access_path_to_dict,
    _dict_to_access_paths,
    _is_access_path_valid,
    _update_nested_dicts,
    get_paths,
)


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
        da: DocList[Inner]

    doc = Outer(
        img=ImageDoc(),
        middle=Middle(img=ImageDoc(), inner=Inner(img=ImageDoc())),
        da=DocList[Inner]([Inner(img=ImageDoc(url='test.png'))]),
    )
    return doc


def test_is_access_path_valid(nested_doc):
    assert _is_access_path_valid(nested_doc.__class__, 'img')
    assert _is_access_path_valid(nested_doc.__class__, 'middle__img')
    assert _is_access_path_valid(nested_doc.__class__, 'middle__inner__img')
    assert _is_access_path_valid(nested_doc.__class__, 'middle')
    assert _is_access_path_valid(nested_doc.__class__, 'da__img__url')


def test_is_access_path_not_valid(nested_doc):
    assert not _is_access_path_valid(nested_doc.__class__, 'inner')
    assert not _is_access_path_valid(nested_doc.__class__, 'some__other__path')
    assert not _is_access_path_valid(nested_doc.__class__, 'middle.inner')


def test_get_access_paths():
    class Painting(BaseDoc):
        title: str
        img: ImageDoc

    access_paths = Painting._get_access_paths()
    assert access_paths == [
        'id',
        'title',
        'img__id',
        'img__url',
        'img__tensor',
        'img__embedding',
        'img__bytes_',
    ]


def test_dict_to_access_paths():
    d = {
        'a0': {'b0': {'c0': 0}, 'b1': {'c0': 1}},
        'a1': {'b0': {'c0': 2, 'c1': 3}, 'b1': 4},
    }
    casted = _dict_to_access_paths(d)
    assert casted == {
        'a0__b0__c0': 0,
        'a0__b1__c0': 1,
        'a1__b0__c0': 2,
        'a1__b0__c1': 3,
        'a1__b1': 4,
    }


def test_access_path_to_dict():
    access_path = 'a__b__c__d__e'
    value = 1
    result = {'a': {'b': {'c': {'d': {'e': value}}}}}
    assert _access_path_to_dict(access_path, value) == result


def test_access_path_dict_to_nested_dict():
    d = {
        'a0__b0__c0': 0,
        'a0__b1__c0': 1,
        'a1__b0__c0': 2,
        'a1__b0__c1': 3,
        'a1__b1': 4,
    }
    casted = _access_path_dict_to_nested_dict(d)
    assert casted == {
        'a0': {'b0': {'c0': 0}, 'b1': {'c0': 1}},
        'a1': {'b0': {'c0': 2, 'c1': 3}, 'b1': 4},
    }


def test_update_nested_dict():
    d1 = {'text': 'hello', 'image': {'tensor': None}}
    d2 = {'image': {'url': 'some.png'}}

    _update_nested_dicts(d1, d2)
    assert d1 == {'text': 'hello', 'image': {'tensor': None, 'url': 'some.png'}}


def test_get_paths():
    paths = list(get_paths(patterns='*.py'))
    for path in paths:
        assert path.endswith('.py')


def test_get_paths_recursive():
    paths_rec = list(get_paths(patterns='**', recursive=True))
    paths_not_rec = list(get_paths(patterns='**', recursive=False))

    assert len(paths_rec) > len(paths_not_rec)


def test_get_paths_exclude():
    paths = list(get_paths(patterns='*.py'))
    paths_wo_init = list(get_paths(patterns='*.py', exclude_regex='__init__.[a-z]*'))

    assert len(paths_wo_init) <= len(paths)
    assert '__init__.py' not in paths_wo_init


def test_shallow_copy():
    from torch import rand

    from docarray import BaseDoc
    from docarray.helper import _shallow_copy_doc
    from docarray.typing import TorchTensor, VideoUrl

    class VideoDoc(BaseDoc):
        url: VideoUrl
        tensor_video: TorchTensor

    class MyDoc(BaseDoc):
        docs: DocList[VideoDoc]
        tensor: TorchTensor

    doc_ori = MyDoc(
        docs=DocList[VideoDoc](
            [
                VideoDoc(
                    url=f'http://example.ai/videos/{i}',
                    tensor_video=rand(256),
                )
                for i in range(10)
            ]
        ),
        tensor=rand(256),
    )

    doc_copy = _shallow_copy_doc(doc_ori)

    assert doc_copy == doc_ori
