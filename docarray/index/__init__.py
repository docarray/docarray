from typing import TYPE_CHECKING

from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.index.backends.hnswlib import HnswDocumentIndex  # noqa: F401

__all__ = []


def __getattr__(name: str):
    if name == 'HnswDocumentIndex':
        import_library('hnswlib', raise_error=True)
        from docarray.index.backends.hnswlib import HnswDocumentIndex  # noqa

        if name not in __all__:
            __all__.append(name)

        return HnswDocumentIndex
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )
