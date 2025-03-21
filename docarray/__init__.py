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
__version__ = '0.40.1'

import logging

from docarray.array import DocList, DocVec
from docarray.base_doc.doc import BaseDoc
from docarray.utils._internal.misc import _get_path_from_docarray_root_level
from docarray.utils._internal.pydantic import is_pydantic_v2


def unpickle_doclist(doc_type, b):
    return DocList[doc_type].from_bytes(b, protocol="protobuf")


def unpickle_docvec(doc_type, tensor_type, b):
    return DocVec[doc_type].from_bytes(b, protocol="protobuf", tensor_type=tensor_type)


if is_pydantic_v2:
    # Register the pickle functions
    def register_serializers():
        import copyreg
        from functools import partial

        unpickle_doc_fn = partial(BaseDoc.from_bytes, protocol="protobuf")

        def pickle_doc(doc):
            b = doc.to_bytes(protocol='protobuf')
            return unpickle_doc_fn, (doc.__class__, b)

        # Register BaseDoc serialization
        copyreg.pickle(BaseDoc, pickle_doc)

        # For DocList, we need to hook into __reduce__ since it's a generic

        def pickle_doclist(doc_list):
            b = doc_list.to_bytes(protocol='protobuf')
            doc_type = doc_list.doc_type
            return unpickle_doclist, (doc_type, b)

        # Replace DocList.__reduce__ with a method that returns the correct format
        def doclist_reduce(self):
            return pickle_doclist(self)

        DocList.__reduce__ = doclist_reduce

        # For DocVec, we need to hook into __reduce__ since it's a generic

        def pickle_docvec(doc_vec):
            b = doc_vec.to_bytes(protocol='protobuf')
            doc_type = doc_vec.doc_type
            tensor_type = doc_vec.tensor_type
            return unpickle_docvec, (doc_type, tensor_type, b)

        # Replace DocList.__reduce__ with a method that returns the correct format
        def docvec_reduce(self):
            return pickle_docvec(self)

        DocVec.__reduce__ = docvec_reduce

    register_serializers()

__all__ = ['BaseDoc', 'DocList', 'DocVec']

logger = logging.getLogger('docarray')

handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def __getattr__(name: str):
    if name in ['Document', 'DocumentArray']:
        raise ImportError(
            f'Cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\'.\n'
            f'The object named \'{name}\' does not exist anymore in this version of docarray.\n'
            f'If you still want to use \'{name}\' please downgrade to version <=0.21.0 '
            f'with: `pip install -U docarray==0.21.0`.'
        )
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )
