from starlette.responses import JSONResponse

from docarray import BaseDocument
from docarray.document.io.json import orjson_dumps


class DocumentResponse(JSONResponse):
    def render(self, content: BaseDocument) -> bytes:
        return orjson_dumps(content.dict())
