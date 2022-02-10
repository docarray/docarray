import itertools
from dataclasses import dataclass, asdict, field
from typing import (
    Union,
    Dict,
    Optional,
    TYPE_CHECKING,
    Sequence,
    Generator,
    Iterator,
)
from pqlite import PQLite

from ..base.backend import BaseBackendMixin
from ....helper import dataclass_from_dict

if TYPE_CHECKING:
    from ....types import (
        DocumentArraySourceType,
    )


@dataclass
class PqliteConfig:
    n_dim: int
    metric: str = 'cosine'
    serialize_config: Dict = field(default_factory=dict)
    data_path: Optional[str] = None


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend. """

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[PqliteConfig, Dict]] = None,
        **kwargs,
    ):
        if not config:
            raise ValueError('Config object must be specified')
        elif isinstance(config, dict):
            config = dataclass_from_dict(PqliteConfig, config)

        self._persist = bool(config.data_path)

        if not self._persist:
            from tempfile import TemporaryDirectory

            config.data_path = TemporaryDirectory().name

        self._config = config

        from .helper import OffsetMapping

        config = asdict(config)
        n_dim = config.pop('n_dim')

        self._pqlite = PQLite(n_dim, **config)
        self._offset2ids = OffsetMapping(
            data_path=config['data_path'],
            in_memory=False,
        )
        from ... import DocumentArray
        from .... import Document

        if _docs is None:
            return

        self.clear()

        if isinstance(
            _docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            self.extend(_docs)
        elif isinstance(_docs, Document):
            self.append(_docs)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_pqlite']
        del state['_offset2ids']
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        config = state['_config']
        config = asdict(config)
        n_dim = config.pop('n_dim')

        from pqlite import PQLite
        from .helper import OffsetMapping

        self._pqlite = PQLite(n_dim, **config)
        self._offset2ids = OffsetMapping(
            data_path=config['data_path'],
            in_memory=False,
        )

    def _get_storage_infos(self) -> Dict:
        storage_infos = super()._get_storage_infos()
        return {
            'Backend': 'PQLite (https://github.com/jina-ai/pqlite)',
            'Distance Metric': self._pqlite.metric.name,
            'Data Path': self._config.data_path,
            'Serialization Protocol': self._config.serialize_config.get('protocol'),
            **storage_infos,
        }
