from .base import BaseDocumentArray
from .mixins import AllMixins
from .storage.memory import StorageMixins


class DocumentArrayMemory(StorageMixins, AllMixins, BaseDocumentArray):
    ...
