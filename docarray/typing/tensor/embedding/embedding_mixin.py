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
from abc import ABC
from typing import Any, Optional, Tuple, Type

from docarray.typing.tensor.abstract_tensor import AbstractTensor


class EmbeddingMixin(AbstractTensor, ABC):
    alternative_type: Optional[Type] = None

    @classmethod
    def __docarray_validate_getitem__(cls, item: Any) -> Tuple[int]:
        shape = super().__docarray_validate_getitem__(item)
        if len(shape) > 1:
            error_msg = f'`{cls}` can only have a single dimension/axis.'
            if cls.alternative_type:
                error_msg += f' Consider using {cls.alternative_type} instead.'
            raise ValueError(error_msg)
        return shape
