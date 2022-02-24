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
from docarray.array.storage.qdrant.helper import QdrantStorageHelper


class GetSetDelMixin(BaseGetSetDelMixin):
    @property
    def client(self) -> QdrantClient:
        raise NotImplementedError()

    @property
    def serialization_config(self) -> dict:
        raise NotImplementedError()

    @property
    def n_dim(self) -> int:
        raise NotImplementedError()

    @property
    def collection_name(self) -> str:
        raise NotImplementedError()

    @property
    def scroll_batch_size(self) -> int:
        raise NotImplementedError()

    def _upload_batch(self, docs: Iterable['Document']):
        self.client.http.points_api.upsert_points(
            name=self.collection_name,
            wait=True,
            point_insert_operations=PointsList(
                points=[self._document_to_qdrant(doc) for doc in docs]
            ),
        )

    def _qdrant_to_document(self, qdrant_record: dict) -> 'Document':
        return Document.from_base64(
            qdrant_record['_serialized'].value[0], **self.serialization_config
        )

    def _document_to_qdrant(self, doc: 'Document') -> 'PointStruct':
        return PointStruct(
            id=self._qmap(doc.id),
            payload=dict(_serialized=doc.to_base64(**self.serialization_config)),
            vector=QdrantStorageHelper.embedding_to_array(doc.embedding, self.n_dim),
        )

    def _get_doc_by_id(self, _id: str) -> 'Document':
        try:
            resp = self.client.http.points_api.get_point(
                name=self.collection_name, id=self._qmap(_id)
            )
            return self._qdrant_to_document(resp.result.payload)
        except UnexpectedResponse as response_error:
            if response_error.status_code in [404, 400]:
                raise KeyError(_id)

    def _del_doc_by_id(self, _id: str):
        self.client.http.points_api.delete_points(
            name=self.collection_name,
            wait=True,
            points_selector=PointIdsList(points=[self._qmap(_id)]),
        )

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        if _id != value.id:
            self._del_doc_by_id(_id)
        self.client.http.points_api.upsert_points(
            name=self.collection_name,
            wait=True,
            point_insert_operations=PointsList(
                points=[self._document_to_qdrant(value)]
            ),
        )

    def scan(self) -> Iterator['Document']:
        offset = None
        while True:
            response = self.client.http.points_api.scroll_points(
                name=self.collection_name,
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
