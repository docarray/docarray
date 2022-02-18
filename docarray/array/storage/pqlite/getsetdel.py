from typing import Iterable

from .helper import OffsetMapping
from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
from ...memory import DocumentArrayInMemory
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    # essential methods start

    def _get_doc_by_id(self, _id: str) -> 'Document':
        doc = self._pqlite.get_doc_by_id(_id)
        if doc is None:
            raise KeyError(f'Can not find Document with id=`{_id}`')
        return doc

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        self._to_numpy_embedding(value)
        docs = DocumentArrayInMemory([value])
        self._pqlite.update(docs)

    def _del_doc_by_id(self, _id: str):
        self._pqlite.delete([_id])

    def _clear_storage(self):
        self._pqlite.clear()

    def _set_docs_by_ids(self, ids, docs: Iterable['Document']):
        docs = DocumentArrayInMemory(docs)
        for doc in docs:
            self._to_numpy_embedding(doc)
        self._pqlite.update(docs)

    def _del_docs_by_ids(self, ids):
        self._pqlite.delete(ids)

    def _load_offset2ids(self):
        self._offsetmapping = OffsetMapping(
            data_path=self._config.data_path, in_memory=False
        )
        self._offsetmapping.create_table()
        self._offset2ids = Offset2ID(self._offsetmapping.get_all_ids())

    def _save_offset2ids(self):
        self._offsetmapping.drop()
        self._offsetmapping.create_table()
        self._offsetmapping._insert(
            [(i, doc_id) for i, doc_id in enumerate(self._offset2ids.ids)]
        )
