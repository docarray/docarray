from .backend import MemoryBackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin


class MemoryStorageMixins(MemoryBackendMixin, GetSetDelMixin, SequenceLikeMixin):
    ...
