import copy
import uuid
from dataclasses import dataclass, field
import warnings

from typing import (
    Dict,
    Optional,
    TYPE_CHECKING,
    Union,
    List,
    Iterable,
    Any,
    Tuple,
    Mapping,
)

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk

from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray import Document
from docarray.helper import dataclass_from_dict, _safe_cast_int

if TYPE_CHECKING:
    from docarray.typing import (
        DocumentArraySourceType,
    )
    from docarray.typing import DocumentArraySourceType, ArrayType


@dataclass
class ElasticConfig:
    n_dim: int  # dims  in elastic
    distance: str = 'cosine'  # similarity in elastic
    hosts: Union[
        str, List[Union[str, Mapping[str, Union[str, int]]]], None
    ] = 'http://localhost:9200'
    index_name: Optional[str] = None
    es_config: Dict[str, Any] = field(default_factory=dict)
    index_text: bool = False
    tag_indices: List[str] = field(default_factory=list)
    batch_size: int = 64
    ef_construction: Optional[int] = None
    m: Optional[int] = None
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None


_banned_indexname_chars = ['[', ' ', '"', '*', '\\', '<', '|', ',', '>', '/', '?', ']']


def _sanitize_index_name(name):
    new_name = name
    for char in _banned_indexname_chars:
        new_name = new_name.replace(char, '')
    return new_name


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    TYPE_MAP = {
        'str': TypeMap(type='text', converter=str),
        'float': TypeMap(type='float', converter=float),
        'int': TypeMap(type='integer', converter=_safe_cast_int),
        'double': TypeMap(type='double', converter=float),
        'long': TypeMap(type='long', converter=_safe_cast_int),
        'bool': TypeMap(type='boolean', converter=bool),
    }

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[ElasticConfig, Dict]] = None,
        **kwargs,
    ):

        config = copy.deepcopy(config)
        if not config:
            raise ValueError('Empty config is not allowed for Elastic storage')
        elif isinstance(config, dict):
            config = dataclass_from_dict(ElasticConfig, config)

        if config.index_name is None:
            id = uuid.uuid4().hex
            config.index_name = 'index_name__' + id

        self._index_name_offset2id = 'offset2id__' + config.index_name
        self._config = config

        self._config.columns = self._normalize_columns(self._config.columns)

        self.n_dim = self._config.n_dim
        self._client = self._build_client()
        self._build_offset2id_index()

        # Note super()._init_storage() calls _load_offset2ids which calls _get_offset2ids_meta
        super()._init_storage()

        if _docs is None:
            return
        elif isinstance(_docs, Iterable):
            self.extend(_docs)
        else:
            if isinstance(_docs, Document):
                self.append(_docs)

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        if 'index_name' not in config_subindex:
            unique_index_name = _sanitize_index_name(
                config_joined['index_name'] + '_subindex_' + subindex_name
            )
            config_joined['index_name'] = unique_index_name
        return config_joined

    def _build_offset2id_index(self):
        if not self._client.indices.exists(index=self._index_name_offset2id):
            self._client.indices.create(index=self._index_name_offset2id, ignore=[404])

    def _build_schema_from_elastic_config(self, elastic_config):
        da_schema = {
            'mappings': {
                'dynamic': 'true',
                '_source': {'enabled': 'true'},
                'properties': {
                    'embedding': {
                        'type': 'dense_vector',
                        'dims': elastic_config.n_dim,
                        'index': 'true',
                        'similarity': elastic_config.distance,
                    },
                    'text': {'type': 'text', 'index': elastic_config.index_text},
                },
            }
        }
        if elastic_config.tag_indices:
            for index in elastic_config.tag_indices:
                da_schema['mappings']['properties'][index] = {
                    'type': 'text',
                    'index': True,
                }

        for col, coltype in self._config.columns.items():
            da_schema['mappings']['properties'][col] = {
                'type': self._map_type(coltype),
                'index': True,
            }

        if self._config.m or self._config.ef_construction:
            index_options = {
                'type': 'hnsw',
                'm': self._config.m or 16,
                'ef_construction': self._config.ef_construction or 100,
            }
            da_schema['mappings']['properties']['embedding'][
                'index_options'
            ] = index_options
        return da_schema

    def _build_client(self):

        client = Elasticsearch(
            hosts=self._config.hosts,
            **self._config.es_config,
        )

        schema = self._build_schema_from_elastic_config(self._config)

        if not client.indices.exists(index=self._config.index_name):
            client.indices.create(
                index=self._config.index_name, mappings=schema['mappings']
            )

        client.indices.refresh(index=self._config.index_name)
        return client

    def _send_requests(self, request, **kwargs) -> List[Dict]:
        """Send bulk request to Elastic and gather the successful info"""

        # for backward compatibility
        if 'chunk_size' not in kwargs:
            kwargs['chunk_size'] = self._config.batch_size

        accumulated_info = []
        for success, info in parallel_bulk(
            self._client,
            request,
            raise_on_error=False,
            raise_on_exception=False,
            **kwargs,
        ):
            if not success:
                warnings.warn(str(info))
            else:
                accumulated_info.append(info)

        return accumulated_info

    def _refresh(self, index_name):
        self._client.indices.refresh(index=index_name)

    def _doc_id_exists(self, doc_id):
        return self._client.exists(index=self._config.index_name, id=doc_id)

    def _update_offset2ids_meta(self):
        """Update the offset2ids in elastic"""
        if self._client.indices.exists(index=self._index_name_offset2id):
            requests = [
                {
                    '_op_type': 'index',
                    '_id': offset_,  # note offset goes here because it's what we want to get by
                    '_index': self._index_name_offset2id,
                    'blob': f'{id_}',
                }  # id here
                for offset_, id_ in enumerate(self._offset2ids.ids)
            ]
            self._send_requests(requests)
            self._client.indices.refresh(index=self._index_name_offset2id)

            # Clean trailing unused offsets
            offset_count = self._client.count(index=self._index_name_offset2id)
            unused_offsets = range(len(self._offset2ids.ids), offset_count['count'])

            if len(unused_offsets) > 0:
                requests = [
                    {
                        '_op_type': 'delete',
                        '_id': offset_,  # note offset goes here because it's what we want to get by
                        '_index': self._index_name_offset2id,
                    }
                    for offset_ in unused_offsets
                ]
                self._send_requests(requests)
                self._client.indices.refresh(index=self._index_name_offset2id)

    def _get_offset2ids_meta(self) -> List:
        """Return the offset2ids stored in elastic

        :return: a list containing ids

        :raises ValueError: error is raised if index _client is not found or no offsets are found
        """
        if not self._client:
            raise ValueError('Elastic client does not exist')

        n_docs = self._client.count(index=self._index_name_offset2id)["count"]

        if n_docs != 0:
            offsets = [x for x in range(n_docs)]
            resp = self._client.mget(index=self._index_name_offset2id, ids=offsets)
            ids = [x['_source']['blob'] for x in resp['docs']]
            return ids
        else:
            return []

    def _map_embedding(self, embedding: 'ArrayType') -> List[float]:
        from docarray.math.helper import EPSILON

        if embedding is None:
            embedding = np.zeros(self.n_dim) + EPSILON
        else:
            from docarray.math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()

        if np.all(embedding == 0):
            embedding = embedding + EPSILON

        return embedding  # .tolist()

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_client']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._client = self._build_client()
