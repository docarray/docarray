from abc import ABC

from docarray.array.storage.weaviate.backend import BackendMixin, WeaviateConfig
from docarray.array.storage.weaviate.find import FindMixin
from docarray.array.storage.weaviate.getsetdel import GetSetDelMixin
from docarray.array.storage.weaviate.seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'WeaviateConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
