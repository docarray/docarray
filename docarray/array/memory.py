from .document import DocumentArray
from .storage.memory import StorageMixins


class DocumentArrayInMemory(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
