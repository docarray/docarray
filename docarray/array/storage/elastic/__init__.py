from abc import ABC

from docarray.array.storage.elastic.backend import BackendMixin, ElasticConfig
from docarray.array.storage.elastic.find import FindMixin
from docarray.array.storage.elastic.getsetdel import GetSetDelMixin
from docarray.array.storage.elastic.seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'ElasticConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
