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
from docarray import BaseDoc, DocList
from docarray.utils._internal.pydantic import is_pydantic_v2


@pytest.mark.skipif(
    is_pydantic_v2,
    reason="Subscripted generics cannot be used with class and instance checks",
)
def test_instance_and_equivalence():
    class MyDoc(BaseDoc):
        text: str

    docs = DocList[MyDoc]([MyDoc(text='hello')])

    assert issubclass(DocList[MyDoc], DocList[MyDoc])
    assert issubclass(docs.__class__, DocList[MyDoc])

    assert isinstance(docs, DocList[MyDoc])


@pytest.mark.skipif(
    is_pydantic_v2,
    reason="Subscripted generics cannot be used with class and instance checks",
)
def test_subclassing():
    class MyDoc(BaseDoc):
        text: str

    class MyDocList(DocList[MyDoc]):
        pass

    docs = MyDocList([MyDoc(text='hello')])

    assert issubclass(MyDocList, DocList[MyDoc])
    assert issubclass(docs.__class__, DocList[MyDoc])

    assert isinstance(docs, MyDocList)
    assert isinstance(docs, DocList[MyDoc])

    assert issubclass(MyDoc, BaseDoc)
    assert not issubclass(DocList[MyDoc], DocList[BaseDoc])
    assert not issubclass(MyDocList, DocList[BaseDoc])
