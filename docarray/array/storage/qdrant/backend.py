import copy
import uuid
from abc import abstractmethod
from dataclasses import dataclass, field, asdict
from typing import (
    Optional,
    TYPE_CHECKING,
    Union,
    Dict,
    Iterable,
    List,
    Tuple,
)

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models.models import (
    Distance,
    CreateCollection,
    PointsList,
    PointStruct,
    HnswConfigDiff,
    VectorParams,
)

from docarray import Document
from docarray.array.storage.base.backend import BaseBackendMixin, TypeMap
from docarray.array.storage.qdrant.helper import DISTANCES
from docarray.helper import dataclass_from_dict, random_identity
from docarray.math.helper import EPSILON

if TYPE_CHECKING:  # pragma: no cover
    from docarray.typing import DocumentArraySourceType, ArrayType


@dataclass
class QdrantConfig:
    n_dim: int
    distance: str = 'cosine'
    collection_name: Optional[str] = None
    list_like: bool = True
    host: Optional[str] = field(default="localhost")
    port: Optional[int] = field(default=6333)
    grpc_port: Optional[int] = field(default=6334)
    prefer_grpc: Optional[bool] = field(default=False)
    api_key: Optional[str] = field(default=None)
    https: Optional[bool] = field(default=None)
    serialize_config: Dict = field(default_factory=dict)
    scroll_batch_size: int = 64
    ef_construct: Optional[int] = None
    full_scan_threshold: Optional[int] = None
    m: Optional[int] = None
    columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None
    root_id: bool = True


class BackendMixin(BaseBackendMixin):
    @property
    @abstractmethod
    def client(self) -> 'QdrantClient':
        raise NotImplementedError()

    @property
    @abstractmethod
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def distance(self) -> 'Distance':
        raise NotImplementedError()

    @classmethod
    def _tmp_collection_name(cls) -> str:
        return uuid.uuid4().hex

    TYPE_MAP = {
        'int': TypeMap(type='integer', converter=int),
        'float': TypeMap(type='float', converter=float),
        'bool': TypeMap(type='int', converter=bool),
        'str': TypeMap(type='keyword', converter=str),
        'text': TypeMap(type='text', converter=str),
        'geo': TypeMap(type='geo', converter=dict),
    }

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[QdrantConfig, Dict]] = None,
        **kwargs,
    ):
        """Initialize qdrant storage.

        :param docs: the list of documents to initialize to
        :param config: the config object used to initialize connection to qdrant server
        :param kwargs: extra keyword arguments
        :raises ValueError: only one of name or docs can be used for initialization,
            raise an error if both are provided
        """
        config = copy.deepcopy(config)
        self._schemas = None

        if not config:
            raise ValueError('Empty config is not allowed for Qdrant storage')
        elif isinstance(config, dict):
            config = dataclass_from_dict(QdrantConfig, config)

        if config.distance not in DISTANCES.keys():
            raise ValueError(
                f'Invalid distance parameter, must be one of: {", ".join(DISTANCES.keys())}'
            )

        if not config.collection_name:
            config.collection_name = self._tmp_collection_name()

        self._n_dim = config.n_dim
        self._distance = config.distance
        self._serialize_config = config.serialize_config

        self._client = QdrantClient(
            host=config.host,
            port=config.port,
            prefer_grpc=config.prefer_grpc,
            grpc_port=config.grpc_port,
            api_key=config.api_key,
            https=config.https,
        )

        self._config = config
        self._list_like = config.list_like
        self._config.columns = self._normalize_columns(self._config.columns)

        self._config.collection_name = (
            self.__class__.__name__ + random_identity()
            if self._config.collection_name is None
            else self._config.collection_name
        )

        self._initialize_qdrant_schema()

        super()._init_storage(**kwargs)

        if docs is None and config.collection_name:
            return

        # To align with Sqlite behavior; if `docs` is not `None` and table name
        # is provided, :class:`DocumentArraySqlite` will clear the existing
        # table and load the given `docs`
        self.clear()
        if isinstance(docs, Iterable):
            self.extend(docs)
        elif isinstance(docs, Document):
            self.append(docs)

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

    def _initialize_qdrant_schema(self):
        if not self._collection_exists(self.collection_name):
            hnsw_config = HnswConfigDiff(
                ef_construct=self._config.ef_construct,
                full_scan_threshold=self._config.full_scan_threshold,
                m=self._config.m,
            )
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.n_dim,
                    distance=self.distance,
                ),
                hnsw_config=hnsw_config,
            )

            for col, coltype in self._config.columns.items():
                if coltype == 'text':
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=col,
                        field_schema=models.TextIndexParams(
                            type="text",
                            tokenizer=models.TokenizerType.WORD,
                        ),
                    )
                else:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=col,
                        field_schema=self._map_type(coltype),
                    )

    def _collection_exists(self, collection_name):
        resp = self.client.get_collections()
        collections = [collection.name for collection in resp.collections]
        return collection_name in collections

    @staticmethod
    def _map_id(doc_id: str):
        # if doc_id is a random ID in hex format, just translate back to UUID str
        # otherwise, create UUID5 from doc_id
        try:
            return str(uuid.UUID(hex=doc_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_client']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._client = QdrantClient(
            host=state['_config'].host,
            port=state['_config'].port,
            prefer_grpc=state['_config'].prefer_grpc,
            grpc_port=state['_config'].grpc_port,
            api_key=state['_config'].api_key,
            https=state['_config'].https,
        )

    def _get_offset2ids_meta(self) -> List[str]:
        if not self._collection_exists(self.collection_name_meta):
            return []
        return self.client.retrieve(self.collection_name_meta, ids=[1])[0].payload.get(
            'offset2id', []
        )

    def _update_offset2ids_meta(self):
        if not self._collection_exists(self.collection_name_meta):
            self.client.recreate_collection(
                collection_name=self.collection_name_meta,
                vectors_config={},  # no vectors
            )

        self.client.upsert(
            collection_name=self.collection_name_meta,
            points=[
                PointStruct(
                    id=1, payload={"offset2id": self._offset2ids.ids}, vector={}
                )
            ],
            wait=True,
        )

    def _map_embedding(self, embedding: 'ArrayType') -> List[float]:
        if embedding is None:
            embedding = np.random.rand(self.n_dim)
        else:
            from docarray.math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

        if embedding.ndim > 1:
            embedding = np.asarray(embedding).squeeze()

        if embedding.ndim == 0:  # scalar
            embedding = np.array([embedding])

        if np.all(embedding == 0):
            embedding = embedding + EPSILON

        return embedding.astype(float).tolist()
