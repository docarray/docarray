from abc import ABC

from docarray.array.storage.memory.backend import BackendMixin
from docarray.array.storage.memory.find import FindMixin
from docarray.array.storage.memory.getsetdel import GetSetDelMixin
from docarray.array.storage.memory.seqlike import SequenceLikeMixin

__all__ = ['StorageMixins']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
