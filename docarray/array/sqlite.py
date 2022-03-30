from .document import DocumentArray

from .storage.sqlite import StorageMixins, SqliteConfig

__all__ = ['SqliteConfig', 'DocumentArraySqlite']


class DocumentArraySqlite(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
