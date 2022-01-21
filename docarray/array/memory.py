from .base import DocumentArray
from .storage.memory import StorageMixins


class DocumentArrayInMemory(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
"""
    def __delitem__(self):
        pass
    def __getitem__(self, item):
        print('getting item')
        pass
    def __setitem__(self):
        pass"""