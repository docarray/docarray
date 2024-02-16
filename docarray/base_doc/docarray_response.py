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
from typing import TYPE_CHECKING, Any

from docarray.base_doc.io.json import orjson_dumps
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from fastapi.responses import JSONResponse
else:
    fastapi = import_library('fastapi', raise_error=True)
    JSONResponse = fastapi.responses.JSONResponse


class DocArrayResponse(JSONResponse):
    """
    This is a custom Response class for FastAPI and starlette. This is needed
    to handle serialization of the Document types when using FastAPI

    ---

    ```python
    from docarray.documets import Text
    from docarray.base_doc import DocResponse


    @app.post("/doc/", response_model=Text, response_class=DocResponse)
    async def create_item(doc: Text) -> Text:
        return doc
    ```

    ---

    """

    def render(self, content: Any) -> bytes:
        return orjson_dumps(content)
