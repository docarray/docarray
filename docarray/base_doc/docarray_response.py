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
