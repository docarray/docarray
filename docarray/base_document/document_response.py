from typing import Any

try:
    from fastapi.responses import JSONResponse, Response
except ImportError:

    class NoImportResponse:
        def __init__(self, *args, **kwargs):
            ImportError('fastapi is not installed')

    Response = JSONResponse = NoImportResponse  # type: ignore


class DocumentResponse(JSONResponse):
    """
    This is a custom Response class for FastAPI and starlette. This is needed
    to handle serialization of the Document types when using FastAPI

      EXAMPLE USAGE
        .. code-block:: python
            from docarray.documets import Text
            from docarray.base_document import DocumentResponse


            @app.post("/doc/", response_model=Text, response_class=DocumentResponse)
            async def create_item(doc: Text) -> Text:
                return doc
    """

    def render(self, content: Any) -> bytes:
        if isinstance(content, bytes):
            return content
        else:
            raise ValueError(f'{self.__class__} only work with json bytes content')
