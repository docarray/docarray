from typing import Sequence, Iterable

from ..base.getsetdel import BaseGetSetDelMixin
from .... import Document


class GetSetDelMixin(BaseGetSetDelMixin):
    """Implement required and derived functions that power `getitem`, `setitem`, `delitem`"""

    # essential methods start

    def _del_doc_by_id(self, _id: str):
        self._sql(f'DELETE FROM {self._table_name} WHERE doc_id=?', (_id,))
        self._commit()

    def _del_doc_by_offset(self, offset: int):
        self._sql(f'DELETE FROM {self._table_name} WHERE item_order=?', (offset,))
        self._commit()

    def _set_doc_by_offset(self, offset: int, value: 'Document'):
        self._sql(
            f'UPDATE {self._table_name} SET serialized_value=? WHERE item_order=?',
            (value, offset),
        )
        self._commit()

    def _set_doc_by_id(self, _id: str, value: 'Document'):
        self._sql(
            f'UPDATE {self._table_name} SET serialized_value=?, doc_id=? WHERE doc_id=?',
            (value, value.id, _id),
        )
        self._commit()

    def _get_doc_by_offset(self, index: int) -> 'Document':
        r = self._sql(
            f'SELECT serialized_value FROM {self._table_name} WHERE item_order = ?',
            (index + (len(self) if index < 0 else 0),),
        )
        res = r.fetchone()
        if res is None:
            raise IndexError('index out of range')
        return res[0]

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
        l = len(self)
        offsets = [o + (l if o < 0 else 0) for o in offsets]
        r = self._sql(
            f"SELECT serialized_value FROM {self._table_name} WHERE item_order in ({','.join(['?'] * len(offsets))})",
            offsets,
        )
        for rr in r:
            yield rr[0]

    def _get_docs_by_slice(self, _slice: slice) -> Iterable['Document']:
        return self._get_docs_by_offsets(range(len(self))[_slice])

    def _del_all_docs(self):
        self._sql(f'DELETE FROM {self._table_name}')
        self._commit()

    def _del_docs_by_slice(self, _slice: slice):
        offsets = range(len(self))[_slice]
        self._sql(
            f"DELETE FROM {self._table_name} WHERE item_order in ({','.join(['?'] * len(offsets))})",
            offsets,
        )
        self._commit()
