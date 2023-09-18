import types
from typing import TYPE_CHECKING

from docarray.store.file import FileDocStore
from docarray.utils._internal.misc import (
    _get_path_from_docarray_root_level,
    import_library,
)

if TYPE_CHECKING:
    from docarray.store.s3 import S3DocStore  # noqa: F401

__all__ = ['FileDocStore']


def __getattr__(name: str):
    lib: types.ModuleType
    if name == 'S3DocStore':
        import_library('smart_open', raise_error=True)
        import_library('botocore', raise_error=True)
        import_library('boto3', raise_error=True)
        import docarray.store.s3 as lib
    else:
        raise ImportError(
            f'cannot import name \'{name}\' from \'{_get_path_from_docarray_root_level(__file__)}\''
        )

    store_cls = getattr(lib, name)

    if name not in __all__:
        __all__.append(name)

    return store_cls
