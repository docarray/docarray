from .document import DocumentArray
from .storage.annlite import StorageMixins, AnnliteConfig

__all__ = ['AnnliteConfig', 'DocumentArrayAnnlite']


class DocumentArrayAnnlite(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
