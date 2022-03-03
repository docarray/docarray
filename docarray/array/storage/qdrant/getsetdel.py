from abc import abstractmethod
from typing import Iterable, Iterator

from qdrant_client import QdrantClient
from qdrant_openapi_client.exceptions import UnexpectedResponse
from qdrant_openapi_client.models.models import (
    PointIdsList,
    PointsList,
    ScrollRequest,
    PointStruct,
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
                self.client.http.points_api.upsert_points(
                    collection_name=self.collection_name,
                    wait=True,
                    point_insert_operations=PointsList(points=batch),
                )
                batch = []
        if len(batch) > 0:
            self.client.http.points_api.upsert_points(
                collection_name=self.collection_name,
                wait=True,
                point_insert_operations=PointsList(points=batch),
            )

    def _qdrant_to_document(self, qdrant_record: dict) -> 'Document':
        return Document.from_base64(
            qdrant_record['_serialized'].value[0], **self.serialization_config
        )

    def _document_to_qdrant(self, doc: 'Document') -> 'PointStruct':
        return PointStruct(
            id=self._map_id(doc.id),
            payload=dict(_serialized=doc.to_base64(**self.serialization_config)),
            vector=self._map_embedding(doc.embedding),
        )

    def _get_doc_by_id(self, _id: str) -> 'Document':
        try:
            resp = self.client.http.points_api.get_point(
                collection_name=self.collection_name, id=self._map_id(_id)
            )
            return self._qdrant_to_document(resp.result.payload)
        except UnexpectedResponse as response_error:
            if response_error.status_code in [404, 400]:
                raise KeyError(_id)

    def _del_doc_by_id(self, _id: str):
        self.client.http.points_api.delete_points(
            collection_name=self.collection_name,
            wait=True,
            points_selector=PointIdsList(points=[self._map_id(_id)]),
        )

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if _id != value.id:
            self._del_doc_by_id(_id)
        self.client.http.points_api.upsert_points(
            collection_name=self.collection_name,
            wait=True,
            point_insert_operations=PointsList(
                points=[self._document_to_qdrant(value)]
            ),
        )

    def scan(self) -> Iterator['Document']:
        offset = None
        while True:
            response = self.client.http.points_api.scroll_points(
                collection_name=self.collection_name,
                scroll_request=ScrollRequest(
                    offset=offset,
                    limit=self.scroll_batch_size,
                    with_payload=['_serialized'],
                    with_vector=False,
                ),
            )
            for point in response.result.points:
                yield self._qdrant_to_document(point.payload)

            if response.result.next_page_offset:
                offset = response.result.next_page_offset
            else:
                break

    def _load_offset2ids(self):
        ids = self._get_offset2ids_meta()
        self._offset2ids = Offset2ID(ids)

    def _save_offset2ids(self):
        self._update_offset2ids_meta()

    def _clear_storage(self):
        self._client.recreate_collection(
            self.collection_name,
            vector_size=self.n_dim,
            distance=self.distance,
        )
