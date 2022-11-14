from abc import abstractmethod
from typing import Iterable, Iterator

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models.models import (
    PointIdsList,
    PointStruct,
    VectorParams,
)

from docarray import Document
from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray.array.storage.base.helper import Offset2ID


class GetSetDelMixin(BaseGetSetDelMixin):
    @property
    @abstractmethod
    def client(self) -> QdrantClient:
        raise NotImplementedError()

    @property
    @abstractmethod
    def serialization_config(self) -> dict:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_dim(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def scroll_batch_size(self) -> int:
        raise NotImplementedError()

    def _upload_batch(self, docs: Iterable['Document']):
        batch = []
        for doc in docs:
            batch.append(self._document_to_qdrant(doc))
            if len(batch) > self.scroll_batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )
                batch = []
        if len(batch) > 0:
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=batch,
            )

    def _qdrant_to_document(self, qdrant_record: dict) -> 'Document':
        return Document.from_base64(
            qdrant_record['_serialized'], **self.serialization_config
        )

    def _document_to_qdrant(self, doc: 'Document') -> 'PointStruct':
        extra_columns = {
            col: doc.tags.get(col) for col, _ in self._config.columns.items()
        }

        return PointStruct(
            id=self._map_id(doc.id),
            payload=dict(
                _serialized=doc.to_base64(**self.serialization_config), **extra_columns
            ),
            vector=self._map_embedding(doc.embedding),
        )

    def _get_doc_by_id(self, _id: str) -> 'Document':
        try:
            resp = self.client.retrieve(
                collection_name=self.collection_name, ids=[self._map_id(_id)]
            )
            if len(resp) == 0:
                raise KeyError(_id)
            return self._qdrant_to_document(resp[0].payload)
        except UnexpectedResponse as response_error:
            if response_error.status_code in [404, 400]:
                raise KeyError(_id)

    def _del_doc_by_id(self, _id: str):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[self._map_id(_id)]),
            wait=True,
        )

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if _id != value.id:
            self._del_doc_by_id(_id)
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=[self._document_to_qdrant(value)],
        )

    def scan(self) -> Iterator['Document']:
        offset = None
        while True:
            response, next_page = self.client.scroll(
                collection_name=self.collection_name,
                offset=offset,
                limit=self.scroll_batch_size,
                with_payload=['_serialized'],
                with_vectors=False,
            )
            for point in response:
                yield self._qdrant_to_document(point.payload)

            if next_page:
                offset = next_page
            else:
                break

    def _load_offset2ids(self):
        if self._list_like:
            ids = self._get_offset2ids_meta()
            self._offset2ids = Offset2ID(ids, list_like=self._list_like)
        else:
            self._offset2ids = Offset2ID([], list_like=self._list_like)

    def _save_offset2ids(self):
        if self._list_like:
            self._update_offset2ids_meta()

    def _clear_storage(self):
        self.client.recreate_collection(
            self.collection_name,
            vectors_config=VectorParams(
                size=self.n_dim,
                distance=self.distance,
            ),
        )
