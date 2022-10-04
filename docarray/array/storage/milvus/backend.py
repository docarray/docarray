import copy
import uuid
from typing import Optional, TYPE_CHECKING, Union, Dict
from dataclasses import dataclass, field
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema

from docarray.array.storage.base.backend import BaseBackendMixin
from docarray.helper import dataclass_from_dict

if TYPE_CHECKING:
    from docarray.typing import (
        DocumentArraySourceType,
    )


@dataclass
class MilvusConfig:
    n_dim: int
    collection_name: str = None
    host: str = 'localhost'
    port: Optional[Union[str, int]] = None  # 19530 for gRPC, 9091 for HTTP
    distance: str = 'IP'  # metric_type in milvus
    index_type: str = 'HNSW'
    index_config: Dict = None  # passed to milvus at index creation time
    collection_config: Dict = field(
        default_factory=dict
    )  # passed to milvus at collection creation time


class BackendMixin(BaseBackendMixin):
    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[MilvusConfig, Dict]] = None,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        if not config:
            raise ValueError('Empty config is not allowed for Elastic storage')
        elif isinstance(config, dict):
            config = dataclass_from_dict(MilvusConfig, config)

        if config.collection_name is None:
            id = uuid.uuid4().hex
            config.index_name = 'docarray__' + id
        self._config = config

        self._connection_alias = 'docarray_default'
        connections.connect(
            alias=self._connection_alias, host=config.host, port=config.port
        )

        self._collection = self._create_collection()

        super()._init_storage(_docs, config, **kwargs)

    def _create_collection(self):
        document_id = FieldSchema(name='document_id', dtype=DataType.STRING)
        order = FieldSchema(name='order', dtype=DataType.STRING, is_primary=True)
        embedding = FieldSchema(
            name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self._config.n_dim
        )
        serialized = FieldSchema(
            name='serialized', dtype=DataType.VARCHAR, max_length=65_535
        )  # this is the maximus allowed length in milvus, could be optimized

        schema = CollectionSchema(
            fields=[document_id, order, embedding, serialized],
            description='docarray collection',
        )
        return Collection(
            name=self._config.collection_name,
            schema=schema,
            using=self._connection_alias,
            **self._config.collection_config,
        )

    def _ensure_unique_config(
        self,
        config_root: dict,
        config_subindex: dict,
        config_joined: dict,
        subindex_name: str,
    ) -> dict:
        if 'collection_name' not in config_subindex:
            config_joined['collection_name'] = (
                config_joined['collection_name'] + '_subindex_' + subindex_name
            )
        return config_joined
