from .document import DocumentArray

from .storage.milvus import StorageMixins, MilvusConfig

__all__ = ['MilvusConfig', 'DocumentArrayMilvus']


class DocumentArrayMilvus(StorageMixins, DocumentArray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)
