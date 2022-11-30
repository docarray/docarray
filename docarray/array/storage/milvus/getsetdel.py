from typing import Iterable, Dict, TYPE_CHECKING

import numpy as np

from docarray import DocumentArray
from docarray.array.storage.base.getsetdel import BaseGetSetDelMixin
from docarray.array.storage.base.helper import Offset2ID
from docarray.array.storage.milvus.backend import (
    _always_true_expr,
    _ids_to_milvus_expr,
    _batch_list,
)

if TYPE_CHECKING:
    from docarray import Document, DocumentArray


class GetSetDelMixin(BaseGetSetDelMixin):
    def _get_doc_by_id(self, _id: str) -> 'Document':
        # to be implemented
        return self._get_docs_by_ids([_id])[0]

    def _del_doc_by_id(self, _id: str):
        # to be implemented
        self._del_docs_by_ids([_id])

    def _set_doc_by_id(self, _id: str, value: 'Document', **kwargs):
        # to be implemented
        self._set_docs_by_ids([_id], [value], None, **kwargs)

    def _load_offset2ids(self):
        if self._list_like:
            collection = self._offset2id_collection
            kwargs = self._update_kwargs_from_config('consistency_level', **dict())
            with self.loaded_collection(collection):
                res = collection.query(
                    expr=_always_true_expr('document_id'),
                    output_fields=['offset', 'document_id'],
                    **kwargs,
                )
            sorted_res = sorted(res, key=lambda k: int(k['offset']))
            self._offset2ids = Offset2ID([r['document_id'] for r in sorted_res])
        else:
            self._offset2ids = Offset2ID([], list_like=self._list_like)

    def _save_offset2ids(self):
        if self._list_like:
            # delete old entries
            self._clear_offset2ids_milvus()
            # insert current entries
            ids = self._offset2ids.ids
            if not ids:
                return
            offsets = [str(i) for i in range(len(ids))]
            dummy_vectors = [np.zeros(1) for _ in range(len(ids))]
            collection = self._offset2id_collection
            collection.insert([offsets, ids, dummy_vectors])

    def _get_docs_by_ids(self, ids: 'Iterable[str]', **kwargs) -> 'DocumentArray':
        if not ids:
            return DocumentArray()
        ids = list(ids)
        kwargs = self._update_kwargs_from_config('consistency_level', **kwargs)
        kwargs = self._update_kwargs_from_config('batch_size', **kwargs)
        with self.loaded_collection():
            docs = DocumentArray()
            for id_batch in _batch_list(ids, kwargs['batch_size']):
                res = self._collection.query(
                    expr=f'document_id in {_ids_to_milvus_expr(id_batch)}',
                    output_fields=['serialized'],
                    **kwargs,
                )
            if not res:
                raise KeyError(f'No documents found for ids {ids}')
            docs.extend(self._docs_from_query_response(res))
        # sort output docs according to input id sorting
        return DocumentArray([docs[d] for d in ids])

    def _del_docs_by_ids(self, ids: 'Iterable[str]', **kwargs) -> 'DocumentArray':
        kwargs = self._update_kwargs_from_config('consistency_level', **kwargs)
        kwargs = self._update_kwargs_from_config('batch_size', **kwargs)
        for id_batch in _batch_list(list(ids), kwargs['batch_size']):
            self._collection.delete(
                expr=f'document_id in {_ids_to_milvus_expr(id_batch)}', **kwargs
            )

    def _set_docs_by_ids(
        self, ids, docs: 'Iterable[Document]', mismatch_ids: 'Dict', **kwargs
    ):
        kwargs = self._update_kwargs_from_config('consistency_level', **kwargs)
        kwargs = self._update_kwargs_from_config('batch_size', **kwargs)
        # delete old entries
        for id_batch in _batch_list(list(ids), kwargs['batch_size']):
            self._collection.delete(
                expr=f'document_id in {_ids_to_milvus_expr(id_batch)}',
                **kwargs,
            )
        for docs_batch in _batch_list(list(docs), kwargs['batch_size']):
            # insert new entries
            payload = self._docs_to_milvus_payload(docs_batch)
            self._collection.insert(payload, **kwargs)

    def _clear_storage(self):
        self._collection.drop()
        self._create_or_reuse_collection()
        self._clear_offset2ids_milvus()

    def _clear_offset2ids_milvus(self):
        self._offset2id_collection.drop()
        self._create_or_reuse_offset2id_collection()
