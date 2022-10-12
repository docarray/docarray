import copy
import uuid
from typing import Optional, TYPE_CHECKING, Union, Dict, Iterable, List, Tuple
from dataclasses import dataclass, field

import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    DataType,
    CollectionSchema,
    has_collection,
)

from docarray import Document, DocumentArray
from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray.helper import dataclass_from_dict, _safe_cast_int

if TYPE_CHECKING:
    from docarray.typing import (
        DocumentArraySourceType,
    )


def always_true_expr(primary_key: str) -> str:
    """
    Returns a Milvus expression that is always true, thus allowing for the retrieval of all entries in a Collection
    Assumes that the primary key is of type DataType.VARCHAR

    :param primary_key: the name of the primary key
    :return: a Milvus expression that is always true for that primary key
    """
    return f'({primary_key} in ["1"]) or ({primary_key} not in ["1"])'


def ids_to_milvus_expr(ids):
    ids = ['"' + _id + '"' for _id in ids]
    return '[' + ','.join(ids) + ']'


@dataclass
class MilvusConfig:
    n_dim: int
    collection_name: str = None
    host: str = 'localhost'
    port: Optional[Union[str, int]] = 19530  # 19530 for gRPC, 9091 for HTTP
    distance: str = 'IP'  # metric_type in milvus
    index_type: str = 'HNSW'
    index_params: Dict = field(
        default_factory=lambda: {
            'M': 4,
            'efConstruction': 200,
        }  # TODO(johannes) check if these defaults are reasonable
    )  # passed to milvus at index creation time. The default assumes 'HNSW' index type
    collection_config: Dict = field(
        default_factory=dict
    )  # passed to milvus at collection creation time
    serialize_config: Dict = field(default_factory=dict)
    consistency_level: str = 'Session'
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None


class BackendMixin(BaseBackendMixin):

    TYPE_MAP = {
        'str': TypeMap(type=DataType.STRING, converter=str),
        'float': TypeMap(type=DataType.FLOAT, converter=float),
        'double': TypeMap(type=DataType.DOUBLE, converter=float),
        'int': TypeMap(type=DataType.INT64, converter=_safe_cast_int),
        'bool': TypeMap(type=DataType.BOOL, converter=bool),
    }

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[MilvusConfig, Dict]] = None,
        **kwargs,
    ):
        config = copy.deepcopy(config)
        if not config:
            raise ValueError('Empty config is not allowed for Milvus storage')
        elif isinstance(config, dict):
            config = dataclass_from_dict(MilvusConfig, config)

        if config.collection_name is None:
            id = uuid.uuid4().hex
            config.collection_name = 'docarray__' + id
        self._config = config
        self._config.columns = self._normalize_columns(self._config.columns)

        self._connection_alias = f'docarray_{config.host}_{config.port}'
        connections.connect(
            alias=self._connection_alias, host=config.host, port=config.port
        )

        self._collection = self._create_or_reuse_collection()
        self._offset2id_collection = self._create_or_reuse_offset2id_collection()
        self._build_index()

        super()._init_storage(_docs, config, **kwargs)

        # To align with Sqlite behavior; if `docs` is not `None` and table name
        # is provided, :class:`DocumentArraySqlite` will clear the existing
        # table and load the given `docs`
        if _docs is None:
            return
        elif isinstance(_docs, Iterable):
            self.clear()
            self.extend(_docs)
        else:
            self.clear()
            if isinstance(_docs, Document):
                self.append(_docs)

    def _create_or_reuse_collection(self):
        if has_collection(self._config.collection_name, using=self._connection_alias):
            return Collection(
                self._config.collection_name, using=self._connection_alias
            )

        document_id = FieldSchema(
            name='document_id', dtype=DataType.VARCHAR, max_length=1024, is_primary=True
        )  # TODO(johannes) this max_length is completely arbitrary
        embedding = FieldSchema(
            name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self._config.n_dim
        )
        serialized = FieldSchema(
            name='serialized', dtype=DataType.VARCHAR, max_length=65_535
        )  # TODO(johannes) this is the maximus allowed length in milvus, could be optimized

        additional_columns = [
            FieldSchema(name=col, dtype=self._map_type(coltype))
            for col, coltype in self._config.columns.items()
        ]

        schema = CollectionSchema(
            fields=[document_id, embedding, serialized, *additional_columns],
            description='DocumentArray collection',
        )
        return Collection(
            name=self._config.collection_name,
            schema=schema,
            using=self._connection_alias,
            **self._config.collection_config,
        )

    def _build_index(self):
        index_params = {
            'metric_type': self._config.distance,
            'index_type': self._config.index_type,
            'params': self._config.index_params,
        }
        self._collection.create_index(field_name='embedding', index_params=index_params)

    def _create_or_reuse_offset2id_collection(self):
        if has_collection(
            self._config.collection_name + '_offset2id', using=self._connection_alias
        ):
            return Collection(
                self._config.collection_name + '_offset2id',
                using=self._connection_alias,
            )

        document_id = FieldSchema(
            name='document_id', dtype=DataType.VARCHAR, max_length=1024
        )  # TODO(johannes) this max_length is completely arbitrary
        offset = FieldSchema(
            name='offset', dtype=DataType.VARCHAR, max_length=1024, is_primary=True
        )  # TODO(johannes) this max_length is completely arbitrary
        dummy_vector = FieldSchema(
            name='dummy_vector', dtype=DataType.FLOAT_VECTOR, dim=1
        )

        schema = CollectionSchema(
            fields=[offset, document_id, dummy_vector],
            description='offset2id for DocumentArray',
        )

        return Collection(
            name=self._config.collection_name + '_offset2id',
            schema=schema,
            using=self._connection_alias,
            # **self._config.collection_config,  # we probably don't want to apply the same config here
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

    def _doc_to_milvus_payload(self, doc):
        return self._docs_to_milvus_payload([doc])

    def _docs_to_milvus_payload(self, docs: 'Iterable[Document]'):
        extra_columns = [
            [self._map_column(doc.tags.get(col), col_type) for doc in docs]
            for col, col_type in self._config.columns.items()
        ]
        return [
            [doc.id for doc in docs],
            [
                doc.embedding
                if doc.embedding is not None
                else np.zeros(self._config.n_dim)
                for doc in docs
            ],
            [doc.to_base64(**self._config.serialize_config) for doc in docs],
            *extra_columns,
        ]

    @staticmethod
    def _docs_from_query_respone(response):
        return DocumentArray([Document.from_base64(d['serialized']) for d in response])

    @staticmethod
    def _docs_from_search_response(
        responses,
    ) -> 'Union[List[DocumentArray], DocumentArray]':
        das = []
        for r in responses:
            das.append(
                DocumentArray(
                    [Document.from_base64(hit.entity.get('serialized')) for hit in r]
                )
            )
        return das if len(das) > 0 else das[0]
