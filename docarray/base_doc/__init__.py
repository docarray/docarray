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
from docarray.base_doc.any_doc import AnyDoc
from docarray.base_doc.base_node import BaseNode
from docarray.base_doc.doc import BaseDoc
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

__all__ = ['AnyDoc', 'BaseDoc', 'BaseNode']


def __getattr__(name: str):
    if name == 'DocArrayResponse':
        import_library('fastapi', raise_error=True)
        from docarray.base_doc.docarray_response import DocArrayResponse

        if name not in __all__:
            __all__.append(name)

        return DocArrayResponse
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )
