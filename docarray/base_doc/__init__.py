from docarray.base_doc.any_doc import AnyDoc
from docarray.base_doc.base_node import BaseNode
from docarray.base_doc.doc import BaseDoc
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

__all__ = ['AnyDoc', 'BaseDoc', 'BaseNode']


def __getattr__(name: str):
    if name == 'DocArrayResponse':
        import_library('fastapi', raise_error=True)
        from docarray.base_doc.docarray_response import DocArrayResponse

        if name not in __all__:
            __all__.append(name)

        return DocArrayResponse
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )
