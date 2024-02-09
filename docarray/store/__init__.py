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
import types
from typing import TYPE_CHECKING

from docarray.store.file import FileDocStore
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.store.s3 import S3DocStore  # noqa: F401

__all__ = ['FileDocStore']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'S3DocStore':
        import_library('smart_open', raise_error=True)
        import_library('botocore', raise_error=True)
        import_library('boto3', raise_error=True)
        import docarray.store.s3 as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    store_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return store_cls
