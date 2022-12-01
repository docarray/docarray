import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from docarray import Document
from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray.helper import dataclass_from_dict, random_identity, filter_dict

from redis import Redis
from redis.commands.search.field import NumericField, TextField, VectorField, GeoField
from redis.commands.search.indexDefinition import IndexDefinition

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import ArrayType, DocumentArraySourceType


@dataclass
class RedisConfig:
    n_dim: int
    host: str = field(default='localhost')
    port: int = field(default=6379)
    index_name: Optional[str] = None
    list_like: bool = True
    update_schema: bool = field(default=True)
    distance: str = field(default='COSINE')
    redis_config: Dict[str, Any] = field(default_factory=dict)
    index_text: bool = field(default=False)
    tag_indices: List[str] = field(default_factory=list)
    batch_size: int = field(default=64)
    method: str = field(default='HNSW')
    ef_construction: Optional[int] = None
    m: Optional[int] = None
    ef_runtime: Optional[int] = None
    block_size: Optional[int] = None
    initial_cap: Optional[int] = None
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    root_id: bool = True


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    TYPE_MAP = {
        'str': TypeMap(type='text', converter=TextField),
        'bytes': TypeMap(type='text', converter=TextField),
        'int': TypeMap(type='integer', converter=NumericField),
        'float': TypeMap(type='float', converter=NumericField),
        'double': TypeMap(type='double', converter=NumericField),
        'long': TypeMap(type='long', converter=NumericField),
        'geo': TypeMap(type='geo', converter=GeoField),
    }

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[RedisConfig, Dict]] = None,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        if not config:
            raise ValueError('Empty config is not allowed for Redis storage')
        elif isinstance(config, dict):
            config = dataclass_from_dict(RedisConfig, config)

        if config.distance not in ['L2', 'IP', 'COSINE']:
            raise ValueError(
                f'Expecting distance metric one of COSINE, L2 OR IP, got {config.distance} instead'
            )
        if config.method not in ['HNSW', 'FLAT']:
            raise ValueError(
                f'Expecting search method one of HNSW OR FLAT, got {config.method} instead'
            )

        if config.redis_config.get('decode_responses'):
            config.redis_config['decode_responses'] = False

        if config.index_name is None:
            config.index_name = 'index_name__' + random_identity()

        self._offset2id_key = config.index_name + '__offset2id'
        self._config = config
        self.n_dim = self._config.n_dim
        self._list_like = config.list_like
        self._doc_prefix = config.index_name + ':'
        self._config.columns = self._normalize_columns(self._config.columns)

        self._client = self._build_client()
        self._build_index()

        super()._init_storage(**kwargs)

        if _docs is None:
            return
        elif isinstance(_docs, Iterable):
            self.extend(_docs)
        elif isinstance(_docs, Document):
            self.append(_docs)

    def _build_client(self):
        client = Redis(
            host=self._config.host,
            port=self._config.port,
            **self._config.redis_config,
        )
        return client

    def _build_index(self, rebuild: bool = False):
        if self._config.update_schema or rebuild:
            if self._config.index_name.encode() in self._client.execute_command(
                'FT._LIST'
            ):
                self._client.ft(index_name=self._config.index_name).dropindex()

            schema = self._build_schema_from_redis_config()
            idef = IndexDefinition(prefix=[self._doc_prefix])
            self._client.ft(index_name=self._config.index_name).create_index(
                schema, definition=idef
            )

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        if 'index_name' not in config_subindex:
            config_joined['index_name'] = (
                config_joined['index_name'] + '_subindex_' + subindex_name
            )
        return config_joined

    def _build_schema_from_redis_config(self):
        index_param = {
            'TYPE': 'FLOAT32',
            'DIM': self.n_dim,
            'DISTANCE_METRIC': self._config.distance,
        }

        if self._config.method == 'HNSW':
            index_options = {
                'M': self._config.m,
                'EF_CONSTRUCTION': self._config.ef_construction,
                'EF_RUNTIME': self._config.ef_runtime,
            }
            index_param.update(filter_dict(index_options))

        if self._config.method == 'FLAT':
            index_options = {'BLOCK_SIZE': self._config.block_size}
            index_param.update(filter_dict(index_options))

        if self._config.initial_cap:
            index_param['INITIAL_CAP'] = self._config.initial_cap
        schema = [VectorField('embedding', self._config.method, index_param)]

        if self._config.index_text:
            schema.append(TextField('text'))

        for index in self._config.tag_indices:
            schema.append(TextField(index))

        for col, coltype in self._config.columns.items():
            schema.append(self._map_column(col, coltype))

        return schema

    def _doc_id_exists(self, doc_id):
        return self._client.exists(self._doc_prefix + doc_id)

    def _map_embedding(self, embedding: 'ArrayType') -> bytes:
        if embedding is not None:
            from docarray.math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()
        else:
            embedding = np.zeros(self.n_dim)
        return embedding.astype(np.float32).tobytes()

    def _get_offset2ids_meta(self) -> List[str]:
        if not self._client.exists(self._offset2id_key):
            return []
        ids = self._client.lrange(self._offset2id_key, 0, -1)
        return [id.decode() for id in ids]

    def _update_offset2ids_meta(self):
        """Update the offset2ids in redis"""
        if self._client.exists(self._offset2id_key):
            self._client.delete(self._offset2id_key)
        if len(self._offset2ids.ids) > 0:
            self._client.rpush(self._offset2id_key, *self._offset2ids.ids)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_client']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._client = self._build_client()
