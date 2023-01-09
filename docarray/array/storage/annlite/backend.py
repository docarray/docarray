import copy
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

from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray.helper import dataclass_from_dict, filter_dict, _safe_cast_int

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import DocumentArraySourceType, ArrayType


@dataclass
class AnnliteConfig:
    n_dim: int
    metric: str = 'cosine'
    list_like: bool = True
    serialize_config: Dict = field(default_factory=dict)
    data_path: Optional[str] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None
    max_connection: Optional[int] = None
    n_components: Optional[int] = None
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    root_id: bool = True


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    TYPE_MAP = {
        'str': TypeMap(type='str', converter=str),
        'float': TypeMap(type='float', converter=float),
        'int': TypeMap(type='int', converter=_safe_cast_int),
    }

    def _map_embedding(self, embedding: 'ArrayType') -> 'ArrayType':
        if embedding is None:
            embedding = np.zeros(self.n_dim, dtype=np.float32)
        elif isinstance(embedding, list):
            from docarray.math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()
        return embedding

    def _normalize_columns(self, columns):
        columns = super()._normalize_columns(columns)
        for key in columns.keys():
            columns[key] = self._map_type(columns[key])
        return columns

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        import os

        if 'data_path' not in config_subindex:
            config_joined['data_path'] = os.path.join(
                config_joined['data_path'], 'subindex_' + subindex_name
            )
        return config_joined

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[AnnliteConfig, Dict]] = None,
        subindex_configs: Optional[Dict] = None,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        from docarray import Document

        if not config:
            raise ValueError('Config object must be specified')
        elif isinstance(config, dict):
            config = dataclass_from_dict(AnnliteConfig, config)

        if config.data_path is None:
            from tempfile import TemporaryDirectory

            config.data_path = TemporaryDirectory().name

        self._config = config
        self._config.columns = self._normalize_columns(self._config.columns)
        config = asdict(config)
        self.n_dim = config.pop('n_dim')
        self._list_like = config.pop("list_like")
        from annlite import AnnLite

        self._annlite = AnnLite(self.n_dim, lock=False, **filter_dict(config))

        super()._init_storage(**kwargs)

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
