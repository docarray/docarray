from .base import BaseDocumentArray
from .mixins import AllMixins
from .storage.sqlite import StorageMixins


class DocumentArraySqlite(StorageMixins, AllMixins, BaseDocumentArray):
    ...
