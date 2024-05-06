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
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import parse_obj_as

from docarray.typing.abstract_type import AbstractType
from docarray.utils._internal.pydantic import bytes_validator, is_pydantic_v2

if is_pydantic_v2:
    from pydantic_core import core_schema

if TYPE_CHECKING:
    from docarray.proto import NodeProto

    if is_pydantic_v2:
        from pydantic import GetCoreSchemaHandler

T = TypeVar('T', bound='BaseBytes')


class BaseBytes(bytes, AbstractType):
    """
    Bytes type for docarray
    """

    @classmethod
    def _docarray_validate(
        cls: Type[T],
        value: Any,
    ) -> T:
        value = bytes_validator(value)
        return cls(value)

    @classmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        return parse_obj_as(cls, pb_msg)

    def _to_node_protobuf(self: T) -> 'NodeProto':
        from docarray.proto import NodeProto

        return NodeProto(blob=self, type=self._proto_type_name)

    if is_pydantic_v2:

        @classmethod
        @abstractmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: 'GetCoreSchemaHandler'
        ) -> 'core_schema.CoreSchema':
            return core_schema.with_info_after_validator_function(
                cls.validate,
                core_schema.bytes_schema(),
            )
