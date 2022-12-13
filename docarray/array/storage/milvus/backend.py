import copy
import uuid
from typing import Optional, TYPE_CHECKING, Union, Dict, Iterable, List, Tuple
from dataclasses import dataclass, field
import re

import numpy as np
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    DataType,
    CollectionSchema,
    has_collection,
    loading_progress,
)

from docarray import Document, DocumentArray
from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray.helper import dataclass_from_dict, _safe_cast_int
from docarray.score import NamedScore

if TYPE_CHECKING:
    from docarray.typing import (
        DocumentArraySourceType,
    )


ID_VARCHAR_LEN = 1024
SERIALIZED_VARCHAR_LEN = (
    65_535  # 65_535 is the maximum that Milvus allows for a VARCHAR field
)
COLUMN_VARCHAR_LEN = 1024
OFFSET_VARCHAR_LEN = 1024


def _always_true_expr(primary_key: str) -> str:
    """
    Returns a Milvus expression that is always true, thus allowing for the retrieval of all entries in a Collection
    Assumes that the primary key is of type DataType.VARCHAR

    :param primary_key: the name of the primary key
    :return: a Milvus expression that is always true for that primary key
    """
    return f'({primary_key} in ["1"]) or ({primary_key} not in ["1"])'


def _ids_to_milvus_expr(ids):
    ids = ['"' + _id + '"' for _id in ids]
    return '[' + ','.join(ids) + ']'


def _batch_list(l: List, batch_size: int):
    """Iterates over a list in batches of size batch_size"""
    if batch_size < 1:
        yield l
        return
    l_len = len(l)
    for ndx in range(0, l_len, batch_size):
        yield l[ndx : min(ndx + batch_size, l_len)]


def _sanitize_collection_name(name):
    """Removes all chars that are not allowed in a Milvus collection name.
    Thus, it removes all chars that are not alphanumeric or an underscore.

    :param name: the collection name to sanitize
    :return: the sanitized collection name.
    """
    return ''.join(
        re.findall('[a-zA-Z0-9_]', name)
    )  # remove everything that is not a letter, number or underscore


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
        }
    )  # passed to milvus at index creation time. The default assumes 'HNSW' index type
    collection_config: Dict = field(
        default_factory=dict
    )  # passed to milvus at collection creation time
    serialize_config: Dict = field(default_factory=dict)
    consistency_level: str = 'Session'
    batch_size: int = -1
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    list_like: bool = True
    root_id: bool = True


class BackendMixin(BaseBackendMixin):

    TYPE_MAP = {
        'str': TypeMap(type=DataType.VARCHAR, converter=str),
        'float': TypeMap(
            type=DataType.DOUBLE, converter=float
        ),  # it doesn't like DataType.FLOAT type, perhaps because python floats are double precision?
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
        self._list_like = config.list_like
        self._config = config
        self._config.columns = self._normalize_columns(self._config.columns)

        self._connection_alias = (
            f'docarray_{config.host}_{config.port}_{uuid.uuid4().hex}'
        )
        connections.connect(
            alias=self._connection_alias, host=config.host, port=config.port
        )

        self._collection = self._create_or_reuse_collection()
        self._offset2id_collection = self._create_or_reuse_offset2id_collection()
        self._build_index()
        super()._init_storage(**kwargs)

        # To align with Sqlite behavior; if `docs` is not `None` and table name
        # is provided, :class:`DocumentArraySqlite` will clear the existing
        # table and load the given `docs`
        if _docs is None:
            return

        self.clear()
        if isinstance(_docs, Iterable):
            self.extend(_docs)
        else:
            if isinstance(_docs, Document):
                self.append(_docs)

    def _create_or_reuse_collection(self):
        if has_collection(self._config.collection_name, using=self._connection_alias):
            return Collection(
                self._config.collection_name, using=self._connection_alias
            )

        document_id = FieldSchema(
            name='document_id',
            dtype=DataType.VARCHAR,
            max_length=ID_VARCHAR_LEN,
            is_primary=True,
        )
        embedding = FieldSchema(
            name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self._config.n_dim
        )
        serialized = FieldSchema(
            name='serialized', dtype=DataType.VARCHAR, max_length=SERIALIZED_VARCHAR_LEN
        )

        additional_columns = []
        for col, coltype in self._config.columns.items():
            mapped_type = self._map_type(coltype)
            if mapped_type == DataType.VARCHAR:
                field_ = FieldSchema(
                    name=col, dtype=mapped_type, max_length=COLUMN_VARCHAR_LEN
                )
            else:
                field_ = FieldSchema(name=col, dtype=mapped_type)
            additional_columns.append(field_)

        schema = CollectionSchema(
            fields=[document_id, embedding, serialized, *additional_columns],
            description='DocumentArray collection schema',
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
            name='document_id', dtype=DataType.VARCHAR, max_length=ID_VARCHAR_LEN
        )
        offset = FieldSchema(
            name='offset',
            dtype=DataType.VARCHAR,
            max_length=OFFSET_VARCHAR_LEN,
            is_primary=True,
        )
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
            config_joined['collection_name'] = _sanitize_collection_name(
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
            [self._map_embedding(doc.embedding) for doc in docs],
            [doc.to_base64(**self._config.serialize_config) for doc in docs],
            *extra_columns,
        ]

    @staticmethod
    def _docs_from_query_response(response):
        return DocumentArray([Document.from_base64(d['serialized']) for d in response])

    @staticmethod
    def _docs_from_search_response(responses, distance: str) -> 'List[DocumentArray]':
        das = []
        for r in responses:
            da = []
            for hit in r:
                doc = Document.from_base64(hit.entity.get('serialized'))
                doc.scores[distance] = NamedScore(value=hit.score)
                da.append(doc)
            das.append(DocumentArray(da))
        return das

    def _update_kwargs_from_config(self, field_to_update, **kwargs):
        kwargs_field_value = kwargs.get(field_to_update, None)
        config_field_value = getattr(self._config, field_to_update, None)

        if (
            kwargs_field_value is not None or config_field_value is None
        ):  # no need to update
            return kwargs

        kwargs[field_to_update] = config_field_value
        return kwargs

    def _map_embedding(self, embedding):
        if embedding is not None:
            from docarray.math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

            if embedding.ndim > 1:
                embedding = np.asarray(embedding).squeeze()
        else:
            embedding = np.zeros(self._config.n_dim)
        return embedding

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_collection']
        del d['_offset2id_collection']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        connections.connect(
            alias=self._connection_alias, host=self._config.host, port=self._config.port
        )
        self._collection = self._create_or_reuse_collection()
        self._offset2id_collection = self._create_or_reuse_offset2id_collection()

    def __enter__(self):
        _ = super().__enter__()
        self._collection.load()
        self._offset2id_collection.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._collection.release()
        self._offset2id_collection.release()
        super().__exit__(exc_type, exc_val, exc_tb)

    def loaded_collection(self, collection=None):
        """
        Context manager to load a collection and release it after the context is exited.
        If the collection is already loaded when entering, it will not be released while exiting.

        :param collection: the collection to load. If None, the main collection of this indexer is used.
        :return: Context manager for the provided collection.
        """

        class LoadedCollectionManager:
            def __init__(self, coll, connection_alias):
                self._collection = coll
                self._loaded_when_enter = False
                self._connection_alias = connection_alias

            def __enter__(self):
                self._loaded_when_enter = (
                    loading_progress(
                        self._collection.name, using=self._connection_alias
                    )['loading_progress']
                    != '0%'
                )
                self._collection.load()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if not self._loaded_when_enter:
                    self._collection.release()

        return LoadedCollectionManager(
            collection if collection else self._collection, self._connection_alias
        )
