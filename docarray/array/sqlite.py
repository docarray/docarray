from .document import DocumentArray

from .storage.sqlite import StorageMixins, SqliteConfig

__all__ = ['SqliteConfig', 'DocumentArraySqlite']


class DocumentArraySqlite(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """
        Ensures that offset2ids are stored in the db after
        operations in the DocumentArray are performed.
        """
        self._save_offset2ids()
