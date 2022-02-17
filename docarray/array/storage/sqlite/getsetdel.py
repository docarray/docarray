from typing import Sequence, Iterable

from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    # essential methods start

    def _del_doc_by_id(self, _id: str):
        self._sql(f'DELETE FROM {self._table_name} WHERE doc_id=?', (_id,))
        self._commit()

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        self._sql(
            f'UPDATE {self._table_name} SET serialized_value=?, doc_id=? WHERE doc_id=?',
            (value, value.id, _id),
        )
        self._commit()

    def _get_doc_by_id(self, id: str) -> 'Document':
        r = self._sql(
            f'SELECT serialized_value FROM {self._table_name} WHERE doc_id = ?', (id,)
        )
        res = r.fetchone()
        if res is None:
            raise KeyError(f'Can not find Document with id=`{id}`')
        return res[0]

    # essentials end here

    # now start the optimized bulk methods
    def _get_docs_by_offsets(self, offsets: Sequence[int]) -> Iterable['Document']:
        ids = [self._offset2ids.get_id(offset) for offset in offsets]
        return self._get_docs_by_ids(ids)

    def _clear_storage(self):
        self._sql(f'DELETE FROM {self._table_name}')
        self._commit()
