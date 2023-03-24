from typing import Any

# from docarray.utils.misc import import_library

try:
    from fastapi.responses import JSONResponse
except ImportError:

    class NoImportResponse:
        def __init__(self, *args, **kwargs):
            ImportError('fastapi is not installed')

    JSONResponse = NoImportResponse  # type: ignore

# if TYPE_CHECKING:
#     from fastapi.responses import JSONResponse, Response
# else:
#     fastapi = import_library('fastapi', raise_error=False)
#     if fastapi is None:
#         class NoImportResponse:
#             def __init__(self, *args, **kwargs):
#                 ImportError('fastapi is not installed')
#
#
#         Response = JSONResponse = NoImportResponse  # type: ignore
#     else:
#         from fastapi.responses import JSONResponse, Response
#


class DocResponse(JSONResponse):
    """
    This is a custom Response class for FastAPI and starlette. This is needed
    to handle serialization of the Document types when using FastAPI

      EXAMPLE USAGE
        .. code-block:: python
            from docarray.documets import Text
            from docarray.base_doc import DocResponse


            @app.post("/doc/", response_model=Text, response_class=DocResponse)
            async def create_item(doc: Text) -> Text:
                return doc
    """

    def render(self, content: Any) -> bytes:
        if isinstance(content, bytes):
            return content
        else:
            raise ValueError(f'{self.__class__} only work with json bytes content')
