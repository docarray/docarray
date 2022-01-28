from dataclasses import dataclass, asdict
from typing import (
    Union,
    Dict,
    Optional,
    TYPE_CHECKING,
)

from ..base.backend import BaseBackendMixin
from ....helper import dataclass_from_dict

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


@dataclass
class PqliteConfig:
    n_dim: int = 1
    metric: str = 'cosine'
    serialize_protocol: str = 'pickle'
    data_path: Optional[str] = None


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend. """

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[PqliteConfig, Dict]] = None,
    ):
        if not config:
            config = PqliteConfig()
        if isinstance(config, dict):
            config = dataclass_from_dict(PqliteConfig, config)

        self._persist = bool(config.data_path)

        if not self._persist:
            from tempfile import TemporaryDirectory

            config.data_path = TemporaryDirectory().name

        self._config = config

        from pqlite import PQLite
        from .helper import OffsetMapping

        config = asdict(config)
        n_dim = config.pop('n_dim')

        self._pqlite = PQLite(n_dim, **config)
        self._offset2ids = OffsetMapping(
            name='docarray',
            data_path=config['data_path'],
            in_memory=False,
        )

        if docs is not None:
            self.clear()
            self.extend(docs)
