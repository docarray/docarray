from abc import ABC

from .backend import BackendMixin, ElasticConfig
from .getsetdel import GetSetDelMixin
from .find import FindMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'ElasticConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
