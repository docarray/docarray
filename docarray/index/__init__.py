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

from docarray.index.backends.in_memory import InMemoryExactNNIndex
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.index.backends.elastic import ElasticDocIndex  # noqa: F401
    from docarray.index.backends.elasticv7 import ElasticV7DocIndex  # noqa: F401
    from docarray.index.backends.epsilla import EpsillaDocumentIndex  # noqa: F401
    from docarray.index.backends.hnswlib import HnswDocumentIndex  # noqa: F401
    from docarray.index.backends.milvus import MilvusDocumentIndex  # noqa: F401
    from docarray.index.backends.qdrant import QdrantDocumentIndex  # noqa: F401
    from docarray.index.backends.redis import RedisDocumentIndex  # noqa: F401
    from docarray.index.backends.weaviate import WeaviateDocumentIndex  # noqa: F401

__all__ = [
    'InMemoryExactNNIndex',
    'ElasticDocIndex',
    'ElasticV7DocIndex',
    'EpsillaDocumentIndex',
    'QdrantDocumentIndex',
    'WeaviateDocumentIndex',
    'RedisDocumentIndex',
    'MilvusDocumentIndex',
]


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'HnswDocumentIndex':
        import_library('hnswlib', raise_error=True)
        import docarray.index.backends.hnswlib as lib
    elif name == 'ElasticDocIndex':
        import_library('elasticsearch', raise_error=True)
        import docarray.index.backends.elastic as lib
    elif name == 'ElasticV7DocIndex':
        import_library('elasticsearch', raise_error=True)
        import docarray.index.backends.elasticv7 as lib
    elif name == 'EpsillaDocumentIndex':
        import_library('pyepsilla', raise_error=True)
        import docarray.index.backends.epsilla as lib
    elif name == 'QdrantDocumentIndex':
        import_library('qdrant_client', raise_error=True)
        import docarray.index.backends.qdrant as lib
    elif name == 'WeaviateDocumentIndex':
        import_library('weaviate', raise_error=True)
        import docarray.index.backends.weaviate as lib
    elif name == 'MilvusDocumentIndex':
        import_library('pymilvus', raise_error=True)
        import docarray.index.backends.milvus as lib
    elif name == 'RedisDocumentIndex':
        import_library('redis', raise_error=True)
        import docarray.index.backends.redis as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    index_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return index_cls
