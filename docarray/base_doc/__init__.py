from docarray.base_doc.any_doc import AnyDoc
from docarray.base_doc.base_node import BaseNode
from docarray.base_doc.doc import BaseDoc

__all__ = ['AnyDoc', 'BaseDoc', 'BaseNode']

from docarray.utils.misc import import_library


def __getattr__(name: str):
    if name == 'DocResponse':
        import_library('fastapi', raise_error=True)
        from docarray.base_doc.doc_response import DocResponse

        __all__.extend(['DocResponse'])
        return DocResponse
