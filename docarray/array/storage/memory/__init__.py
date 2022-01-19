from .backend import MemoryBackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['MemoryStorageMixins']


class MemoryStorageMixins(MemoryBackendMixin, GetSetDelMixin, SequenceLikeMixin):
    ...
