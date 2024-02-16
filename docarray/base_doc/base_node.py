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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, Optional, Type

if TYPE_CHECKING:
    from docarray.proto import NodeProto

T = TypeVar('T')


class BaseNode(ABC):
    """
    A DocumentNode is an object than can be nested inside a Document.
    A Document itself is a DocumentNode as well as prebuilt type
    """

    _proto_type_name: Optional[str] = None

    @abstractmethod
    def _to_node_protobuf(self) -> 'NodeProto':
        """Convert itself into a NodeProto message. This function should
        be called when the self is nested into another Document that need to be
        converted into a protobuf

        :return: the nested item protobuf message
        """
        ...

    @classmethod
    @abstractmethod
    def from_protobuf(cls: Type[T], pb_msg: T) -> T:
        ...

    def _docarray_to_json_compatible(self):
        """
        Convert itself into a json compatible object
        """
        ...
