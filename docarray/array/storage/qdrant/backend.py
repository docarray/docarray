import itertools
import uuid
from dataclasses import dataclass, field
from typing import (
    Optional,
    TYPE_CHECKING,
    Union,
    Dict,
    Sequence,
    Generator,
    Iterator,
    Iterable,
)

from qdrant_client import QdrantClient
from qdrant_openapi_client.models.models import Distance

from docarray import Document
from docarray.array.storage.base.backend import BaseBackendMixin

if TYPE_CHECKING:
    from docarray.types import (
        DocumentArraySourceType,
    )


@dataclass
class QdrantConfig:
    n_dim: int
    distance: Distance = Distance.COSINE
    collection_name: Optional[str] = None
    connection: Optional[Union[str, QdrantClient]] = field(default="localhost:6333")
    serialize_config: Dict = field(default_factory=dict)
    scroll_batch_size: int = 64


class BackendMixin(BaseBackendMixin):
    def clear(self):
        self._client.recreate_collection(
            self._config.collection_name,
            vector_size=self._config.n_dim,
            distance=self._config.distance,
        )

    def extend(self, docs: Iterable):
        raise NotImplementedError()

    def append(self, doc: Document):
        raise NotImplementedError()

    @classmethod
    def _tmp_collection_name(cls) -> str:
        return uuid.uuid4().hex

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[QdrantConfig, Dict]] = None,
        **kwargs
    ):
        """Initialize qdrant storage.

        :param docs: the list of documents to initialize to
        :param config: the config object used to initialize connection to qdrant server
        :param kwargs: extra keyword arguments
        :raises ValueError: only one of name or docs can be used for initialization,
            raise an error if both are provided
        """

        from ... import DocumentArray

        self._schemas = None

        if not config:
            raise ValueError('Empty config is not allowed for Qdrant storage')

        if not config.collection_name:
            config.collection_name = self._tmp_collection_name()

        self._n_dim = config.n_dim
        self._serialize_config = config.serialize_config

        if isinstance(config.connection, str):
            host, *port = config.connection.split(':')
            if port:
                self._client = QdrantClient(host=host, port=port[0])
        else:
            self._client = config.connection

        self._config = config

        super()._init_storage()

        if docs is None and config.collection_name:
            return

        # To align with Sqlite behavior; if `docs` is not `None` and table name
        # is provided, :class:`DocumentArraySqlite` will clear the existing
        # table and load the given `docs`
        self.clear()
        if isinstance(
            docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
        ):
            self.extend(docs)
        elif isinstance(docs, Document):
            self.append(docs)
