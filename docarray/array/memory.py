from .base import BaseDocumentArray
from .mixins import AllMixins
from .storage.memory import StorageMixins


class DocumentArrayInMemory(StorageMixins, AllMixins, BaseDocumentArray):
    ...
