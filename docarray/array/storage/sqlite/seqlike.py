from typing import Union, Optional, Iterable

from ..base.seqlike import BaseSequenceLikeMixin
from .... import Document
from ...memory import DocumentArrayInMemory


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    def _insert_doc_at_idx(self, doc, idx: Optional[int] = None):
        if idx is None:
            idx = len(self)
        self._sql(
            f'INSERT INTO {self._table_name} (doc_id, serialized_value, item_order) VALUES (?, ?, ?)',
            (doc.id, doc, idx),
        )
        self._offset2ids.insert(idx, doc.id)

    def _shift_index_right_backward(self, start: int):
        idx = len(self) - 1
        while idx >= start:
            self._sql(
                f'UPDATE {self._table_name} SET item_order = ? WHERE item_order = ?',
                (idx + 1, idx),
            )
            idx -= 1

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        length = len(self)
        if index < 0:
            index = length + index
        index = max(0, min(length, index))
        self._shift_index_right_backward(index)
        self._insert_doc_at_idx(doc=value, idx=index)
        self._commit()

    def append(self, doc: 'Document') -> None:
        self._sql(
            f'INSERT INTO {self._table_name} (doc_id, serialized_value, item_order) VALUES (?, ?, ?)',
            (doc.id, doc, len(self)),
        )
        self._offset2ids.append(doc.id)
        self._commit()

    def __del__(self) -> None:
        super().__del__()
        if not self._persist:
            self._sql(
                'DELETE FROM metadata WHERE table_name=? AND container_type=?',
                (self._table_name, self.__class__.__name__),
            )
            self._sql(f'DROP TABLE IF EXISTS {self._table_name}')
            self._commit()

    def __contains__(self, item: Union[str, 'Document']):
        if isinstance(item, str):
            r = self._sql(f'SELECT 1 FROM {self._table_name} WHERE doc_id=?', (item,))
            return len(list(r)) > 0
        elif isinstance(item, Document):
            return item.id in self  # fall back to str check
        else:
            return False

    def __len__(self) -> int:
        request = self._sql(f'SELECT COUNT(*) FROM {self._table_name}')
        return request.fetchone()[0]

    def __repr__(self):
        return f'<DocumentArray[SQLite] (length={len(self)}) at {id(self)}>'

    def __eq__(self, other):
        """In sqlite backend, data are considered as identical if configs point to the same database source"""
        return (
            type(self) is type(other)
            and type(self._config) is type(other._config)
            and self._config == other._config
        )

    def extend(self, docs: Iterable['Document']) -> None:

        self_len = len(self)
        for doc in docs:
            self._sql(
                f'INSERT INTO {self._table_name} (doc_id, serialized_value, item_order) VALUES (?, ?, ?)',
                (doc.id, doc, self_len),
            )
            self._offset2ids.append(doc.id)
            self_len += 1
        self._commit()
