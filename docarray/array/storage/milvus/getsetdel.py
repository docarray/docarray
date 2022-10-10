from typing import Iterable

import numpy as np

from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray import Document
from docarray.array.storage.base.helper import Offset2ID
from docarray.array.storage.milvus.backend import always_true_expr, ids_to_milvus_expr


class GetSetDelMixin(BaseGetSetDelMixin):
    def _get_doc_by_id(self, _id: str) -> 'Document':
        # to be implemented
        return self._get_docs_by_ids([_id])[0]

    def _del_doc_by_id(self, _id: str):
        # to be implemented
        self._del_docs_by_ids([_id])

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        # to be implemented
        self._set_docs_by_ids([_id], [value], None)

    def _load_offset2ids(self):
        collection = self._offset2id_collection
        collection.load()
        res = collection.query(
            expr=always_true_expr('document_id'),
            output_fields=['offset', 'document_id'],
            consistency_level='Strong',
        )
        collection.release()
        sorted_res = sorted(res, key=lambda k: int(k['offset']))
        self._offset2ids = Offset2ID([r['document_id'] for r in sorted_res])

    def _clear_offset2ids_milvus(self):
        collection = self._offset2id_collection
        collection.delete(
            expr=f'document_id in {ids_to_milvus_expr(self._offset2ids.ids)}',
            consistency_level='Strong',
        )

    def _save_offset2ids(self):
        # delete old entries
        self._clear_offset2ids_milvus()
        # insert current entries
        collection = self._offset2id_collection
        ids = self._offset2ids.ids
        offsets = [str(i) for i in range(len(ids))]
        dummy_vectors = [np.zeros(1) for _ in range(len(ids))]
        collection.insert([offsets, ids, dummy_vectors])

    def _get_docs_by_ids(self, ids: 'Iterable[str]') -> 'DocumentArray':
        self._collection.load()
        res = self._collection.query(
            expr=f'document_id in {ids_to_milvus_expr(ids)}',
            output_fields=['serialized'],
            consistency_level='Strong',
        )
        self._collection.release()
        return self._docs_from_milvus_respone(res)

    def _del_docs_by_ids(self, ids: 'Iterable[str]') -> 'DocumentArray':
        self._collection.delete(
            expr=f'document_id in {ids_to_milvus_expr(ids)}', consistency_level='Strong'
        )

    def _set_docs_by_ids(self, ids, docs: 'Iterable[Document]', mismatch_ids: 'Dict'):
        # TODO(johannes) check if deletion is necesarry if ids already match
        # delete old entries
        self._collection.delete(
            expr=f'document_id in {ids_to_milvus_expr(ids)}', consistency_level='Strong'
        )
        # insert new entries
        payload = self._docs_to_milvus_payload(docs)
        self._collection.insert(payload)

    def _clear_storage(self):
        collection = self._collection
        collection.delete(
            expr=f'document_id in {ids_to_milvus_expr(self._offset2ids.ids)}',
            consistency_level='Strong',
        )
        self._clear_offset2ids_milvus()
        self._offset2ids = Offset2ID()
