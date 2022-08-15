from abc import ABC

from docarray.array.storage.annlite.backend import BackendMixin, AnnliteConfig
from docarray.array.storage.annlite.find import FindMixin
from docarray.array.storage.annlite.getsetdel import GetSetDelMixin
from docarray.array.storage.annlite.seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'AnnliteConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
