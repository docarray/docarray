from typing import Iterable, Dict

from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Provide concrete implementation for ``__getitem__``, ``__setitem__``,
    and ``__delitem__`` for ``DocumentArrayWeaviate``"""

    def _getitem(self, wid: str) -> 'Document':
        """Helper method for getting item with weaviate as storage

        :param wid: weaviate id
        :raises KeyError: raise error when weaviate id does not exist in storage
        :return: Document
        """
        try:
            resp = self._client.data_object.get_by_id(wid, with_vector=True)
            return Document.from_base64(
                resp['properties']['_serialized'], **self._serialize_config
            )
        except Exception as ex:
            raise KeyError(wid) from ex

    def _get_doc_by_id(self, _id: str) -> 'Document':
        """Concrete implementation of base class' ``_get_doc_by_id``

        :param _id: the id of the document
        :return: the retrieved document from weaviate
        """
        return self._getitem(self._map_id(_id))

    def _set_doc_by_id(self, _id: str, value: 'Document', flush: bool = True):
        """Concrete implementation of base class' ``_set_doc_by_id``

        :param _id: the id of doc to update
        :param value: the document to update to
        """
        if _id != value.id:
            self._del_doc_by_id(_id)

        payload = self._doc2weaviate_create_payload(value)
        self._client.batch.add_data_object(**payload)
        if flush:
            self._client.batch.flush()

    def _set_docs_by_ids(self, ids, docs: Iterable['Document'], mismatch_ids: Dict):
        """Overridden implementation of _set_docs_by_ids in order to add docs in batches and flush at the end

        :param ids: the ids used for indexing
        """
        for _id, doc in zip(ids, docs):
            self._set_doc_by_id(_id, doc, flush=False)
        self._client.batch.flush()

    def _del_doc_by_id(self, _id: str):
        """Concrete implementation of base class' ``_del_doc_by_id``

        :param _id: the id of the document to delete
        """
        wid = self._map_id(_id)
        if self._client.data_object.exists(wid):
            self._client.data_object.delete(wid)

    def _clear_storage(self):
        """Concrete implementation of base class' ``_clear_storage``"""
        if self._class_name:
            self._client.schema.delete_class(self._class_name)
            self._client.schema.delete_class(self._meta_name)
            self._load_or_create_weaviate_schema()

    def _load_offset2ids(self):
        ids, self._offset2ids_wid = self._get_offset2ids_meta()
        self._offset2ids = Offset2ID(ids)

    def _save_offset2ids(self):
        self._update_offset2ids_meta()
