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
from typing import TYPE_CHECKING

from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from google.protobuf import __version__ as __pb__version__
else:
    protobuf = import_library('google.protobuf', raise_error=True)
    __pb__version__ = protobuf.__version__


if __pb__version__.startswith('4'):
    from docarray.proto.pb.docarray_pb2 import (
        DictOfAnyProto,
        DocListProto,
        DocProto,
        DocVecProto,
        ListOfAnyProto,
        ListOfDocArrayProto,
        ListOfDocVecProto,
        NdArrayProto,
        NodeProto,
    )
else:
    from docarray.proto.pb2.docarray_pb2 import (
        DictOfAnyProto,
        DocListProto,
        DocProto,
        DocVecProto,
        ListOfAnyProto,
        ListOfDocArrayProto,
        ListOfDocVecProto,
        NdArrayProto,
        NodeProto,
    )

__all__ = [
    'DocListProto',
    'DocProto',
    'NdArrayProto',
    'NodeProto',
    'DocVecProto',
    'DocListProto',
    'ListOfDocArrayProto',
    'ListOfDocVecProto',
    'ListOfAnyProto',
    'DictOfAnyProto',
]
