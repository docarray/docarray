import copy
import uuid
from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    TYPE_CHECKING,
    Union,
    List,
    Iterable,
    Any,
)

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..base.backend import BaseBackendMixin
from .... import Document
from ....helper import dataclass_from_dict

if TYPE_CHECKING:
    from ....typing import (
        DocumentArraySourceType,
    )
    from ....typing import DocumentArraySourceType, ArrayType


@dataclass
class ElasticConfig:
    n_dim: int  # dims  in elastic
    distance: str = 'cosine'  # similarity in elastic
    hosts: str = 'http://localhost:9200'
    index_name: Optional[str] = None
    es_config: Dict[str, Any] = field(default_factory=dict)


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

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
            self._persist = False
            id = uuid.uuid4().hex
            config.index_name = 'index_name__' + id
        else:
            self._persist = True

        self._index_name_offset2id = 'offset2id__' + config.index_name
        self._config = config
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

    def _build_offset2id_index(self):
        if not self._client.indices.exists(index=self._index_name_offset2id):
            self._client.indices.create(index=self._index_name_offset2id, ignore=[404])

    def _build_schema_from_elastic_config(self, elastic_config):
        da_schema = {
            "mappings": {
                "dynamic": "true",
                "_source": {"enabled": "true"},
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": elastic_config.n_dim,
                        "index": "true",
                        "similarity": elastic_config.distance,
                    },
                },
            }
        }
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

    def _send_requests(self, request):
        bulk(self._client, request)

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
            r = bulk(self._client, requests)
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
        from ....math.helper import EPSILON

        if embedding is None:
            embedding = np.zeros(self.n_dim) + EPSILON
        else:
            from ....math.ndarray import to_numpy_array

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

    # def clear(self):
    #    self._client.indices.delete(index=self._config.index_name)
