from abc import ABC

from .backend import BackendMixin, ElasticSearchConfig
from .getsetdel import GetSetDelMixin

__all__ = ['StorageMixins', 'ElasticSearchConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
