from abc import ABC
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .... import Document

from .backend import BackendMixin, PqliteConfig
from .find import FindMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'PqliteConfig']


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin, ABC):
    ...
