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
        resp = self._client.data_object.get_by_id(wid, with_vector=True)
        if not resp:
            raise KeyError(wid)
        return Document.from_base64(
            resp['properties']['_serialized'], **self._serialize_config
        )

    def _get_doc_by_id(self, _id: str) -> 'Document':
        """Concrete implementation of base class' ``_get_doc_by_id``

        :param _id: the id of the document
        :return: the retrieved document from weaviate
        """
        return self._getitem(self._wmap(_id))

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        """Concrete implementation of base class' ``_set_doc_by_id``

        :param _id: the id of doc to update
        :param value: the document to update to
        """
        if _id != value.id:
            self._del_doc_by_id(_id)
        wid = self._wmap(value.id)
        payload = self._doc2weaviate_create_payload(value)
        if self._client.data_object.exists(wid):
            self._client.data_object.delete(wid)
        self._client.data_object.create(**payload)

    def _del_doc_by_id(self, _id: str):
        """Concrete implementation of base class' ``_del_doc_by_id``

        :param _id: the id of the document to delete
        """
        wid = self._wmap(_id)
        if self._client.data_object.exists(wid):
            self._client.data_object.delete(wid)

    def _clear_storage(self):
        """ Concrete implementation of base class' ``_clear_storage``"""
        if self._class_name:
            self._client.schema.delete_class(self._class_name)
            self._client.schema.delete_class(self._meta_name)
            self._load_or_create_weaviate_schema()

    def _load_offset2ids(self):
        ids, self._offset2ids_wid = self._get_offset2ids_meta()
        self._offset2ids = Offset2ID(ids)

    def _save_offset2ids(self):
        self._update_offset2ids_meta()
