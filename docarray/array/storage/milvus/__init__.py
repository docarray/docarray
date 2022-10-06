from abc import ABC

from .backend import BackendMixin, MilvusConfig
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'MilvusConfig']


class StorageMixins(BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
