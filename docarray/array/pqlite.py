from .document import DocumentArray
from .storage.pqlite import StorageMixins, PqliteConfig

__all__ = ['PqliteConfig', 'DocumentArrayPqlite']


class DocumentArrayPqlite(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
