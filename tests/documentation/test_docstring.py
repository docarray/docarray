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

SUB_MODULE_TO_CHECK = [
    docarray,
    docarray.index,
    docarray.data,
    docarray.documents,
    docarray.store,
    docarray.typing,
]


def get_obj_to_check(lib):
    obj_to_check = []
    for obj in lib.__all__:
        obj_to_check.append(getattr(lib, obj))
    return obj_to_check


obj_to_check = []

for lib in SUB_MODULE_TO_CHECK:
    obj_to_check.extend(get_obj_to_check(lib))

members = []
for obj in obj_to_check:
    members.extend(get_codeblock_members(obj))

print(f"len(members) = {len(members)}")
print(f"len(set(members)) = {len(set(members))}")


@pytest.mark.parametrize("obj", members, ids=lambda d: d.__qualname__)
def test_member(obj):
    check_docstring(obj)
