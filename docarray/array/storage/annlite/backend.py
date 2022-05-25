from dataclasses import dataclass, asdict, field
from typing import (
    Union,
    Dict,
    Optional,
    TYPE_CHECKING,
    Iterable,
    List,
    Tuple,
)

import numpy as np

from ..base.backend import BaseBackendMixin
from ....helper import dataclass_from_dict, filter_dict

if TYPE_CHECKING:
    from ....typing import DocumentArraySourceType, ArrayType


@dataclass
class AnnliteConfig:
    n_dim: int
    metric: str = 'cosine'
    serialize_config: Dict = field(default_factory=dict)
    data_path: Optional[str] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None
    max_connection: Optional[int] = None
    columns: Optional[List[Tuple[str, str]]] = None


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    TYPE_MAP = {'str': 'TEXT', 'float': 'float', 'int': 'integer'}

    def _map_embedding(self, embedding: 'ArrayType') -> 'ArrayType':
        if embedding is None:
            embedding = np.zeros(self.n_dim, dtype=np.float32)
        elif isinstance(embedding, list):
            from ....math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()
        return embedding

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[AnnliteConfig, Dict]] = None,
        **kwargs,
    ):
        if not config:
            raise ValueError('Config object must be specified')
        elif isinstance(config, dict):
            config = dataclass_from_dict(AnnliteConfig, config)

        self._persist = bool(config.data_path)

        if not self._persist:
            from tempfile import TemporaryDirectory

            config.data_path = TemporaryDirectory().name

        self._config = config

        if self._config.columns is None:
            self._config.columns = []

        for i in range(len(self._config.columns)):
            self._config.columns[i] = (
                self._config.columns[i][0],
                self._map_type(self._config.columns[i][1]),
            )

        config = asdict(config)
        self.n_dim = config.pop('n_dim')

        from annlite import AnnLite

        self._annlite = AnnLite(self.n_dim, lock=False, **filter_dict(config))
        from .... import Document

        super()._init_storage()

        if _docs is None:
            return

        self.clear()

        if isinstance(_docs, Iterable):
            self.extend(_docs)
        elif isinstance(_docs, Document):
            self.append(_docs)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state['_annlite']
        del state['_offsetmapping']
        return state

    def __setstate__(self, state):
        self.__dict__ = state

        config = state['_config']
        config = asdict(config)
        n_dim = config.pop('n_dim')

        from annlite import AnnLite

        self._annlite = AnnLite(n_dim, lock=False, **filter_dict(config))

    def __len__(self):
        return self._annlite.index_size
