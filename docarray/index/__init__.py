import types
from typing import TYPE_CHECKING

from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.index.backends.elastic import ElasticDocIndex  # noqa: F401
    from docarray.index.backends.elasticv7 import ElasticV7DocIndex  # noqa: F401
    from docarray.index.backends.hnswlib import HnswDocumentIndex  # noqa: F401

__all__ = []


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'HnswDocumentIndex':
        import_library('hnswlib', raise_error=True)
        import docarray.index.backends.hnswlib as lib
    elif name == 'ElasticDocIndex':
        import_library('elasticsearch', raise_error=True)
        import docarray.index.backends.elastic as lib
    elif name == 'ElasticV7DocIndex':
        import_library('elasticsearch', raise_error=True)
        from elasticsearch import __version__ as __es__version__

        import docarray.index.backends.elasticv7 as lib

        if __es__version__[0] > 7:
            raise ImportError(
                'ElasticV7DocIndex requires the elasticsearch library to be version 7.10.1'
            )

    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    index_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return index_cls
