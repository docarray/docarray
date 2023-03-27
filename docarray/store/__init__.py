from docarray.store.file import FileDocStore

__all__ = ['FileDocStore']


from docarray.utils.misc import import_library


def __getattr__(name: str):
    if name == 'JACDocStore':
        import_library('hubble', raise_error=True)
        from docarray.store.jac import JACDocStore  # noqa: F401

        __all__.extend(['JACDocStore'])

    elif name == 'S3DocStore':
        import_library('smart_open', raise_error=True)
        import_library('botocore', raise_error=True)
        import_library('boto3', raise_error=True)
        from docarray.store.s3 import S3DocStore  # noqa: F401

        __all__.extend(['S3DocStore'])
