from dataclasses import dataclass, asdict
from typing import (
    Union,
    Dict,
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
    n_dim: int = 1
    metric: str = 'cosine'
    data_path: str = 'data'


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend. """

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[PqliteConfig, Dict]] = None,
    ):
        if not config:
            config = PqliteConfig()
        self._config = config

        from pqlite import PQLite
        from .helper import OffsetMapping

        config = asdict(config)
        n_dim = config.pop('n_dim')

        self._pqlite = PQLite(n_dim, **config)
        self._offset2ids = OffsetMapping(
            name='offset2ids', data_path=config['data_path'], in_memory=True
        )

        if docs is not None:
            self.clear()
            self.extend(docs)
