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
"""
this test check the docstring of all of our public API. It does it
by checking the `__all__` of each of our namespace.

to add a new namespace you need to
* import it
* add it to the `SUB_MODULE_TO_CHECK` list
"""

import pytest
from mktestdocs import check_docstring, get_codeblock_members

import docarray.data
import docarray.documents
import docarray.index
import docarray.store
import docarray.typing
from docarray.utils import filter, find, map

SUB_MODULE_TO_CHECK = [
    docarray,
    docarray.index,
    docarray.data,
    docarray.documents,
    docarray.store,
    docarray.typing,
    find,
    map,
    filter,
]


def get_obj_to_check(lib):
    obj_to_check = []
    all_test = getattr(lib, '__all__')
    try:
        all_test = getattr(lib, '__all_test__')
    except (AttributeError, ImportError):
        pass
    for obj in all_test:
        obj_to_check.append(getattr(lib, obj))
    return obj_to_check


obj_to_check = []

for lib in SUB_MODULE_TO_CHECK:
    obj_to_check.extend(get_obj_to_check(lib))


members = []
for obj in obj_to_check:
    members.extend(get_codeblock_members(obj))


@pytest.mark.parametrize("obj", members, ids=lambda d: d.__qualname__)
def test_member(obj):
    check_docstring(obj)
