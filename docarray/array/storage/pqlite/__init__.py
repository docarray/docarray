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

    def _to_numpy_embedding(self, doc: 'Document'):
        if doc.embedding is None:
            doc.embedding = np.zeros(self._pqlite.dim, dtype=np.float32)
        elif isinstance(doc.embedding, list):
            doc.embedding = np.array(doc.embedding, dtype=np.float32)
