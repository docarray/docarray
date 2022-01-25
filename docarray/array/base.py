from abc import ABC
from docarray.array.mixins import AllMixins


class BaseDocumentArray(ABC):
    def __init__(self, *args, storage: str = 'memory', **kwargs):
        super().__init__()
        self._init_storage(*args, **kwargs)


class DocumentArray(AllMixins, BaseDocumentArray):
    def __new__(cls, *args, storage: str = 'memory', **kwargs):
        if cls is DocumentArray:
            if storage == 'memory':
                from docarray.array.memory import DocumentArrayInMemory

                instance = super().__new__(DocumentArrayInMemory)
            elif storage == 'sqlite':
                from docarray.array.sqlite import DocumentArraySqlite

                instance = super().__new__(DocumentArraySqlite)
            else:
                raise ValueError(f'storage=`{storage}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance
