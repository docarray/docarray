from typing import TYPE_CHECKING

from docarray.store.file import FileDocStore
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:
    from docarray.store.jac import JACDocStore  # noqa: F401
    from docarray.store.s3 import S3DocStore  # noqa: F401

__all__ = ['FileDocStore']


def __getattr__(name: str):
    if name not in __all__:
        __all__.append(name)

    if name == 'JACDocStore':
        import_library('hubble', raise_error=True)
        from docarray.store.jac import JACDocStore  # noqa

        return JACDocStore

    elif name == 'S3DocStore':
        import_library('smart_open', raise_error=True)
        import_library('botocore', raise_error=True)
        import_library('boto3', raise_error=True)
        from docarray.store.s3 import S3DocStore  # noqa

        return S3DocStore
