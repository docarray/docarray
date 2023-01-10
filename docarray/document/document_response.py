try:
    from starlette.responses import JSONResponse
except ImportError:

    class JSONResponse:
        raise ImportError("Starlette is required to use JSONResponse")


from docarray import BaseDocument
from docarray.document.io.json import orjson_dumps


class DocumentResponse(JSONResponse):
    def render(self, content: BaseDocument) -> bytes:
        return orjson_dumps(content.dict())
