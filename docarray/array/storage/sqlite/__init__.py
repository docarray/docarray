from abc import ABC

from .backend import BackendMixin, SqliteConfig
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin
from ..memory.find import FindMixin  # temporary delegate to in-memory find API

__all__ = ['StorageMixins', 'SqliteConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
