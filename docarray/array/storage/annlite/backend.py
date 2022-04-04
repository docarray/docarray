from dataclasses import dataclass, asdict, field
from typing import (
    Union,
    Dict,
    Optional,
    TYPE_CHECKING,
    Iterable,
)

import numpy as np

from ..base.backend import BaseBackendMixin
from ....helper import dataclass_from_dict

if TYPE_CHECKING:
    from ....typing import DocumentArraySourceType, ArrayType


@dataclass
class AnnliteConfig:
    n_dim: int
    metric: str = 'cosine'
    serialize_config: Dict = field(default_factory=dict)
    data_path: Optional[str] = None


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

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

        config = asdict(config)
        self.n_dim = config.pop('n_dim')

        from annlite import AnnLite

        self._annlite = AnnLite(self.n_dim, lock=False, **config)
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

        self._annlite = AnnLite(n_dim, lock=False, **config)
