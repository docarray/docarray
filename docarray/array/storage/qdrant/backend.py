import uuid
from dataclasses import dataclass, field
from typing import (
    Optional,
    TYPE_CHECKING,
    Union,
    Dict,
    Iterable,
    List,
)

import numpy as np
from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import (
    Distance,
    CreateCollection,
    PointsList,
    PointStruct,
)

from docarray import Document
from docarray.array.storage.base.backend import BaseBackendMixin
from docarray.array.storage.qdrant.helper import DISTANCES
from docarray.helper import dataclass_from_dict, random_identity
from docarray.math.helper import EPSILON

if TYPE_CHECKING:
    from ....typing import DocumentArraySourceType, ArrayType


@dataclass
class QdrantConfig:
    n_dim: int
    distance: str = 'cosine'
    collection_name: Optional[str] = None
    host: Optional[str] = field(default="localhost")
    port: Optional[int] = field(default=6333)
    serialize_config: Dict = field(default_factory=dict)
    scroll_batch_size: int = 64


class BackendMixin(BaseBackendMixin):
    @classmethod
    def _tmp_collection_name(cls) -> str:
        return uuid.uuid4().hex

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
        self._serialize_config = config.serialize_config

        self._client = QdrantClient(host=config.host, port=config.port)

        self._config = config
        self._persist = bool(self._config.collection_name)

        self._config.collection_name = (
            self.__class__.__name__ + random_identity()
            if self._config.collection_name is None
            else self._config.collection_name
        )

        self._persist = self._config.collection_name
        self._initialize_qdrant_schema()

        super()._init_storage()

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

    def _initialize_qdrant_schema(self):
        if not self._collection_exists(self.collection_name):
            self.client.http.collections_api.create_collection(
                self.collection_name,
                CreateCollection(vector_size=self.n_dim, distance=self.distance),
            )

    def _collection_exists(self, collection_name):
        resp = self.client.http.collections_api.get_collections()
        collections = [collection.name for collection in resp.result.collections]
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
            host=state['_config'].host, port=state['_config'].port
        )

    def _get_offset2ids_meta(self) -> List[str]:
        if not self._collection_exists(self.collection_name_meta):
            return []
        return (
            self.client.http.points_api.get_point(self.collection_name_meta, id=1)
            .result.payload['offset2id']
            .value
        )

    def _update_offset2ids_meta(self):
        if not self._collection_exists(self.collection_name_meta):
            self.client.http.collections_api.create_collection(
                self.collection_name_meta,
                CreateCollection(vector_size=1, distance=Distance.COSINE),
            )

        self.client.http.points_api.upsert_points(
            collection_name=self.collection_name_meta,
            wait=True,
            point_insert_operations=PointsList(
                points=[
                    PointStruct(
                        id=1, payload={"offset2id": self._offset2ids.ids}, vector=[1]
                    )
                ]
            ),
        )

    def _map_embedding(self, embedding: 'ArrayType') -> List[float]:
        if embedding is None:
            embedding = np.random.rand(self.n_dim)
        else:
            from ....math.ndarray import to_numpy_array

            embedding = to_numpy_array(embedding)

        if embedding.ndim > 1:
            embedding = np.asarray(embedding).squeeze()

        if np.all(embedding == 0):
            embedding = embedding + EPSILON
        return embedding.tolist()
