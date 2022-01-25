from .document import DocumentArray
from .storage.sqlite import StorageMixins


class DocumentArraySqlite(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
