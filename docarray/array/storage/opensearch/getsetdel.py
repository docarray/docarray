from typing import Sequence, Iterable, Dict, List

from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray.array.storage.base.helper import Offset2ID
from docarray import Document
import numpy as np


class GetSetDelMixin(BaseGetSetDelMixin):

    MAX_OPENSEARCH_RETURNED_DOCS = 10000

    def _getitem(self, doc_id: str) -> 'Document':
        """Helper method for getting item with OpenSearch as storage
        :param doc_id:  id of the document
        :raises KeyError: raise error when opensearch id does not exist in storage
        :return: Document
        """
        try:
            result = self._client.get(index=self._config.index_name, id=doc_id)
            doc = Document.from_base64(result['_source']['blob'])
            return doc
        except Exception as ex:
            raise KeyError(doc_id) from ex

    def _get_doc_by_id(self, _id: str) -> 'Document':
        """Concrete implementation of base class' ``_get_doc_by_id``
        :param _id: the id of the document
        :return: the retrieved document from opensearch
        """
        return self._getitem(_id)

    def _get_docs_by_ids(self, ids: Sequence[str]) -> Iterable[Document]:
        """Concrete implementation of base class' ``_get_docs_by_ids``
        :param ids:  ids of the document
        :return: Iterable[Document]
        """
        accumulated_docs = []
        accumulated_docs_id_not_found = []

        if not ids:
            return accumulated_docs

        # Handle if doc len is more than MAX_ES_RETURNED_DOCS
        for pos in range(0, len(ids), self.MAX_OPENSEARCH_RETURNED_DOCS):
            es_docs = self._client.mget(
                body={'ids': ids[pos : pos + self.MAX_OPENSEARCH_RETURNED_DOCS]},
                index=self._config.index_name,
            )['docs']
            for doc in es_docs:
                if doc['found']:
                    accumulated_docs.append(
                        Document.from_base64(doc['_source']['blob'])
                    )
                else:
                    accumulated_docs_id_not_found.append(doc['_id'])

        if accumulated_docs_id_not_found:
            raise KeyError(accumulated_docs_id_not_found, accumulated_docs)

        return accumulated_docs

    def _del_doc_by_id(self, _id: str):
        """Concrete implementation of base class' ``_del_doc_by_id``
        :param _id: the id of the document to delete
        """
        if self._doc_id_exists(_id):
            self._client.delete(index=self._config.index_name, id=_id)
        self._refresh(self._config.index_name)

    def _set_doc_by_id(self, _id: str, value: Document):
        """Concrete implementation of base class' ``_set_doc_by_id``
        :param _id: the id of doc to update
        :param value: the document to update to
        """
        if _id != value.id:
            self._del_doc_by_id(_id)

        request = [self._document_to_opensearch_request(value)]

        self._send_requests(request)
        self._refresh(self._config.index_name)

    def _set_docs_by_ids(self, ids, docs: Iterable[Document], mismatch_ids: Dict):
        """Overridden implementation of _set_docs_by_ids in order to add docs in batches and flush at the end
        :param ids: the ids used for indexing
        """
        for _id, doc in zip(ids, docs):
            self._set_doc_by_id(_id, doc)

        self._refresh(self._config.index_name)

    def _load_offset2ids(self):
        if self._list_like:
            ids = self._get_offset2ids_meta()
            self._offset2ids = Offset2ID(ids, list_like=self._list_like)
        else:
            self._offset2ids = Offset2ID([], list_like=self._list_like)

    def _save_offset2ids(self):
        if self._list_like:
            self._update_offset2ids_meta()

    def _document_to_opensearch_request(self, doc: Document) -> Dict:
        extra_columns = {
            col: doc.tags.get(col) for col, _ in self._config.columns.items()
        }
        request = {
            '_op_type': 'index',
            '_id': doc.id,
            '_index': self._config.index_name,
            'embedding': self._map_embedding(doc.embedding),
            'blob': doc.to_base64(),
            **extra_columns,
        }

        if self._config.tag_indices:
            for index in self._config.tag_indices:
                request[index] = doc.tags.get(index)

        if doc.text:
            request['text'] = doc.text
        return request
