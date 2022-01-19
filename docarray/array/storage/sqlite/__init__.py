from .backend import SqliteBackendMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['SqliteStorageMixins']


class SqliteStorageMixins(SqliteBackendMixin, GetSetDelMixin, SequenceLikeMixin):
    ...
