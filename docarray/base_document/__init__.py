from typing_extensions import TYPE_CHECKING

from docarray.base_document.any_document import AnyDocument
from docarray.base_document.base_node import BaseNode
from docarray.base_document.document import BaseDocument
from docarray.utils.misc import import_library

__all__ = ['AnyDocument', 'BaseDocument', 'BaseNode']


if TYPE_CHECKING:
    from docarray.base_document.document_response import DocumentResponse  # noqa: F401


def __getattr__(name: str):
    if name == 'DocumentResponse':
        import_library('fastapi', raise_error=True)
        from docarray.base_document.document_response import DocumentResponse  # noqa

        __all__.extend(['DocumentResponse'])
        return DocumentResponse
