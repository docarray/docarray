from .document import DocumentArray
from .storage.annlite import StorageMixins, AnnliteConfig

__all__ = ['AnnliteConfig', 'DocumentArrayAnnlite']


class DocumentArrayAnnlite(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """
        Ensures that offset2ids are stored in the db after
        operations in the DocumentArray are performed.
        """
        self._save_offset2ids()
