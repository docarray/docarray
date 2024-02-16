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
from typing import Callable, Dict, Type, TypeVar

from docarray.typing.abstract_type import AbstractType

_PROTO_TYPE_NAME_TO_CLASS: Dict[str, Type[AbstractType]] = {}

T = TypeVar('T', bound='AbstractType')


def _register_proto(
    proto_type_name: str,
) -> Callable[[Type[T]], Type[T]]:
    """Register a new type to be used in the protobuf serialization.

    This will add the type key to the global registry of types key used in the proto
    serialization and deserialization. This is for internal usage only.

    ---

    ```python
    from docarray.typing.proto_register import register_proto
    from docarray.typing.abstract_type import AbstractType


    @register_proto(proto_type_name='my_type')
    class MyType(AbstractType):
        ...
    ```

    ---

    :param cls: the class to register
    :return: the class
    """

    if proto_type_name in _PROTO_TYPE_NAME_TO_CLASS.keys():
        raise ValueError(
            f'the key {proto_type_name} is already registered in the global registry'
        )

    def _register(cls: Type[T]) -> Type[T]:
        cls._proto_type_name = proto_type_name

        _PROTO_TYPE_NAME_TO_CLASS[proto_type_name] = cls
        return cls

    return _register
