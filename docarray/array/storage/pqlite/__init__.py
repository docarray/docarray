from .backend import BackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin
from abc import ABC

__all__ = ['StorageMixins']


class StorageMixins(BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
