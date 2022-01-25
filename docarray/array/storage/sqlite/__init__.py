from abc import ABC

from .backend import BackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins']


class StorageMixins(BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
