import dataclasses
from dataclasses import dataclass
from typing import (
    Optional,
    TYPE_CHECKING,
)

from ..base.backend import BaseBackendMixin

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


@dataclass
class PqliteConfig:
    dim: int = 256
    metric: str = 'cosine'
    data_path: str = 'data'


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend. """

    def _insert_doc_at_idx(self, doc, idx: Optional[int] = None):
        raise NotImplementedError

    def _shift_index_right_backward(self, start: int):
        raise NotImplementedError

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[PqliteConfig] = None,
    ):
        if not config:
            config = PqliteConfig()

        from pqlite import PQLite

        self._pqlite = PQLite(**config)

        if docs is not None:
            self.clear()
            self.extend(docs)
