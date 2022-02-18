from operator import itemgetter
from typing import Sequence, Iterable

from ..base.getsetdel import BaseGetSetDelMixin
from ..base.helper import Offset2ID
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

    def _del_docs_by_ids(self, ids: str) -> Iterable['Document']:
        self._sql(
            f"DELETE FROM {self._table_name} WHERE doc_id in ({','.join(['?'] * len(ids))})",
            ids,
        )
        self._commit()

    def _load_offset2ids(self):
        r = self._sql(
            f"SELECT doc_id FROM {self._table_name} ORDER BY item_order",
        )
        self._offset2ids = Offset2ID(list(map(itemgetter(0), r)))

    def _save_offset2ids(self):
        for offset, doc_id in enumerate(self._offset2ids):
            self._sql(
                f"""
                    UPDATE {self._table_name} SET item_order = ? WHERE {self._table_name}.doc_id = ?
                """,
                (offset, doc_id),
            )
        self._commit()
