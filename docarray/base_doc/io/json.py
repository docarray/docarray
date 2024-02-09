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
from typing import Any, Callable, Dict, Type

import orjson

from docarray.utils._internal.pydantic import is_pydantic_v2

if not is_pydantic_v2:
    from pydantic.json import ENCODERS_BY_TYPE
else:
    ENCODERS_BY_TYPE: Dict[Type[Any], Callable[[Any], Any]] = {
        bytes: lambda o: o.decode(),
        frozenset: list,
        set: list,
    }


def _default_orjson(obj):
    """
    default option for orjson dumps.
    :param obj:
    :return: return a json compatible object
    """
    from docarray.base_doc import BaseNode

    if isinstance(obj, BaseNode):
        return obj._docarray_to_json_compatible()
    else:
        for cls_, encoder in ENCODERS_BY_TYPE.items():
            if isinstance(obj, cls_):
                return encoder(obj)
        return obj


def orjson_dumps(v, *, default=None) -> bytes:
    # dumps to bytes using orjson
    return orjson.dumps(v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY)


def orjson_dumps_and_decode(v, *, default=None) -> str:
    # dumps to str using orjson
    return orjson_dumps(v, default=default).decode()
