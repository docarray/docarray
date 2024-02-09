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

from docarray.utils._internal.query_language.lookup import dunder_get, lookup


class A:
    class B:
        c = 0
        d = 'docarray'
        e = [0, 1]
        f = {}

    b: B = B()


@pytest.mark.parametrize('input', [A(), {'b': {'c': 0, 'd': 'docarray'}}])
def test_dunder_get(input):
    assert dunder_get(input, 'b__c') == 0

    expected_exception = KeyError if isinstance(input, dict) else AttributeError
    with pytest.raises(expected_exception):
        _ = dunder_get(input, 'z')

    with pytest.raises(expected_exception):
        _ = dunder_get(input, 'b__z')


@pytest.mark.parametrize(
    'input', [A(), {'b': {'c': 0, 'd': 'docarray', 'e': [0, 1], 'f': {}}}]
)
def test_lookup(input):
    assert lookup('b__c.exact', 0, input)
    assert not lookup('b__c.gt', 0, input)
    assert lookup('b__c.gte', 0, input)
    assert not lookup('b__c.lt', 0, input)
    assert lookup('b__c.lte', 0, input)
    assert lookup('b__d.regex', 'array*', input)
    assert lookup('b__d.contains', 'array', input)
    assert lookup('b__d.icontains', 'Array', input)
    assert lookup('b__d.in', ['a', 'docarray'], input)
    assert lookup('b__d.nin', ['a', 'b'], input)
    assert lookup('b__d.startswith', 'doc', input)
    assert lookup('b__d.istartswith', 'Doc', input)
    assert lookup('b__d.endswith', 'array', input)
    assert lookup('b__d.iendswith', 'Array', input)
    assert lookup('b__e.size', 2, input)
    assert not lookup('b__e.size', 3, input)
    assert lookup('b__d.size', len('docarray'), input)
    assert not lookup('b__e.size', len('docarray') + 1, input)
    assert not lookup('b__z.exists', True, input)
    assert lookup('b__z.exists', False, input)
    assert not lookup('b__f__z.exists', True, input)
    assert lookup('b__f__z.exists', False, input)
