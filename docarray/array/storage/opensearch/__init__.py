from abc import ABC

from docarray.array.storage.opensearch.backend import BackendMixin, OpenSearchConfig
from docarray.array.storage.opensearch.find import FindMixin
from docarray.array.storage.opensearch.getsetdel import GetSetDelMixin
from docarray.array.storage.opensearch.seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'OpenSearchConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
