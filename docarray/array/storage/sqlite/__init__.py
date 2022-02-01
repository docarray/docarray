from abc import ABC

from .backend import BackendMixin, SqliteConfig
from .binary import SqliteBinaryIOMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'SqliteConfig']


class StorageMixins(
    SqliteBinaryIOMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC
):
    ...
