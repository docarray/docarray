from .backend import SqliteBackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin
from abc import ABC

__all__ = ['SqliteStorageMixins']


class SqliteStorageMixins(SqliteBackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
