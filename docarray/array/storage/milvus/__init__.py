from abc import ABC

from .backend import BackendMixin, MilvusConfig
from .find import FindMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'MilvusConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
