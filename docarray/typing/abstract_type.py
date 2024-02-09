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

from docarray.utils._internal.pydantic import is_pydantic_v2

if TYPE_CHECKING:
    if is_pydantic_v2:
        from pydantic import GetCoreSchemaHandler
        from pydantic_core import core_schema

from docarray.base_doc.base_node import BaseNode

T = TypeVar('T')


class AbstractType(BaseNode):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    @abstractmethod
    def _docarray_validate(cls: Type[T], value: Any) -> T:
        ...

    if is_pydantic_v2:

        @classmethod
        def validate(cls: Type[T], value: Any, _: Any) -> T:
            return cls._docarray_validate(value)

    else:

        @classmethod
        def validate(
            cls: Type[T],
            value: Any,
        ) -> T:
            return cls._docarray_validate(value)

    if is_pydantic_v2:

        @classmethod
        @abstractmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: Any, _handler: 'GetCoreSchemaHandler'
        ) -> 'core_schema.CoreSchema':
            ...
