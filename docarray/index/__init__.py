from typing import TYPE_CHECKING

from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.index.backends.hnswlib import HnswDocumentIndex  # noqa: F401

__all__ = []


def __getattr__(name: str):

    if name not in __all__:
        __all__.append(name)

    if name == 'HnswDocumentIndex':
        import_library('hnswlib', raise_error=True)
        from docarray.index.backends.hnswlib import HnswDocumentIndex  # noqa

        return HnswDocumentIndex
