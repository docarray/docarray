from abc import ABC

from .backend import BackendMixin, WeaviateConfig
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin
from .find import FindMixin

__all__ = ['StorageMixins', 'WeaviateConfig']


class StorageMixins(BackendMixin, GetSetDelMixin, SequenceLikeMixin, FindMixin):
    ...
