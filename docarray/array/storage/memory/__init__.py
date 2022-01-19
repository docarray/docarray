from .backend import MemoryBackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin
from abc import ABC

__all__ = ['MemoryStorageMixins']


class MemoryStorageMixins(MemoryBackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
