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
from pydantic import parse_obj_as

from docarray import BaseDoc
from docarray.documents import TextDoc


def test_simple_init():
    t = TextDoc(text='hello')
    assert t.text == 'hello'


def test_str_init():
    t = parse_obj_as(TextDoc, 'hello')
    assert t.text == 'hello'


def test_doc():
    class MyDoc(BaseDoc):
        text1: TextDoc
        text2: TextDoc

    doc = MyDoc(text1='hello', text2=TextDoc(text='world'))

    assert doc.text1.text == 'hello'
    assert doc.text2.text == 'world'
