from typing import Iterable, TYPE_CHECKING

from .backend import BackendMixin, QdrantConfig
from .find import FindMixin
from .getsetdel import GetSetDelMixin
from .seqlike import SequenceLikeMixin

__all__ = ['StorageMixins', 'QdrantConfig']

if TYPE_CHECKING:
    from qdrant_client import QdrantClient
    from qdrant_openapi_client.models.models import Distance
    from docarray import Document


class StorageMixins(FindMixin, BackendMixin, GetSetDelMixin, SequenceLikeMixin):
    @property
    def serialize_config(self) -> dict:
        return self._config.serialize_config

    @property
    def distance(self) -> 'Distance':
        return self._config.distance

    def __iter__(self) -> Iterable['Document']:
        yield from self.scan()

    def extend(self, docs: Iterable['Document']):
        self._upload_batch(docs)

    def append(self, doc: 'Document'):
        self._set_doc_by_id(doc.id, value=doc)

    @property
    def serialization_config(self) -> dict:
        return self._serialize_config

    @property
    def n_dim(self) -> int:
        return self._n_dim

    @property
    def collection_name(self) -> str:
        return self._config.collection_name

    @property
    def config(self):
        return self._config

    @property
    def client(self) -> 'QdrantClient':
        return self._client

    @property
    def scroll_batch_size(self) -> int:
        return self._config.scroll_batch_size
