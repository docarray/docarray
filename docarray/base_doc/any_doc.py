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
from typing import Type

from docarray.utils._internal.pydantic import is_pydantic_v2

from .doc import BaseDoc


class AnyDoc(BaseDoc):
    """
    AnyDoc is a Document that is not tied to any schema
    """

    class Config:
        _load_extra_fields_from_protobuf = True  # I introduce this variable to allow to load more that the fields defined in the schema
        # will documented this behavior later if this fix our problem

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @classmethod
    def _get_field_annotation(cls, field: str) -> Type['BaseDoc']:
        """
        Accessing the nested python Class define in the schema.
        Could be useful for reconstruction of Document in
        serialization/deserilization
        :param field: name of the field
        :return:
        """
        return AnyDoc

    @classmethod
    def _get_field_annotation_array(cls, field: str) -> Type:
        from docarray import DocList

        return DocList

    if is_pydantic_v2:

        def dict(self, *args, **kwargs):
            raise NotImplementedError(
                "dict() method is not implemented for pydantic v2. Now pydantic requires a schema to dump the dict, but AnyDoc is schemaless"
            )
