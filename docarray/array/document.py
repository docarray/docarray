from .base import BaseDocumentArray
from .mixins import AllMixins
from .storage.memory import StorageMixins


class DocumentArray(StorageMixins, AllMixins, BaseDocumentArray):
    ...
