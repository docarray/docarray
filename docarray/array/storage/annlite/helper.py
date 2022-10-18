from pathlib import Path
from typing import Optional, List, Tuple

from annlite.storage.table import Table


class OffsetMapping(Table):
    def __init__(
        self,
        name: str = 'offset2ids',
        data_path: Optional[Path] = None,
        in_memory: bool = True,
    ):
        super().__init__(name, data_path, in_memory=in_memory)
        self.create_table()
        self._size = None

    def close(self):
        self._conn.close()

    def create_table(self):
        sql = f'''CREATE TABLE IF NOT EXISTS {self.name}
                            (offset INTEGER NOT NULL PRIMARY KEY,
                             doc_id TEXT NOT NULL)'''

        self.execute(sql, commit=True)

    def drop(self):
        sql = f'''DROP TABLE IF EXISTS {self.name}'''
        self.execute(sql, commit=True)

    def clear(self):
        super().clear()
        self._size = None

    def __len__(self):
        return self.size

    @property
    def size(self):
        if self._size is None:
            sql = f'SELECT MAX(offset) from {self.name} LIMIT 1;'
            result = self._conn.execute(sql).fetchone()
            self._size = result[0] + 1 if result[0] else 0

        return self._size

    def extend_doc_ids(self, doc_ids: List[str], commit: bool = True):
        offsets = [self.size + i for i in range(len(doc_ids))]
        offset_ids = list(zip(offsets, doc_ids))
        self._insert(offset_ids, commit=commit)

    def _insert(self, offset_ids: List[Tuple[int, str]], commit: bool = True):
        sql = f'INSERT INTO {self.name}(offset, doc_id) VALUES (?, ?);'
        self.execute_many(sql, offset_ids, commit=commit)
        self._size = self.size + len(offset_ids)

    def get_id_by_offset(self, offset: int):
        offset = len(self) + offset if offset < 0 else offset
        sql = f'SELECT doc_id FROM {self.name} WHERE offset = ? LIMIT 1;'
        result = self._conn.execute(sql, (offset,)).fetchone()
        return str(result[0]) if result is not None else None

    def get_ids_by_offsets(self, offsets: List[int]) -> List[str]:
        return [self.get_id_by_offset(offset) for offset in offsets]

    def get_offsets_by_ids(self, ids: List[str]) -> List[int]:
        return [self.get_offset_by_id(k) for k in ids]

    def get_offset_by_id(self, doc_id: str):
        sql = f'SELECT offset FROM {self.name} WHERE doc_id=? LIMIT 1;'
        result = self._conn.execute(sql, (doc_id,)).fetchone()
        return result[0] if result else None

    def get_all_ids(self):
        sql = f'SELECT doc_id FROM {self.name} ORDER BY offset'
        result = self._conn.execute(sql).fetchall()
        return [r[0] for r in result] if result else []

    def del_at_offset(self, offset: int, commit: bool = True):
        offset = len(self) + offset if offset < 0 else offset
        sql = f'DELETE FROM {self.name} WHERE offset=?'
        self._conn.execute(sql, (offset,))
        self.shift_offset(offset, shift_step=1, direction='left', commit=commit)

        self._size -= 1

    def del_at_offsets(self, offsets: List[int], commit: bool = True):
        for offset in sorted(offsets, reverse=True):
            self.del_at_offset(offset, commit=False)
        if commit:
            self.commit()

    def insert_at_offset(self, offset: int, doc_id: str, commit: bool = True):
        offset = len(self) + offset if offset < 0 else offset
        self.shift_offset(offset - 1, shift_step=1, direction='right', commit=False)
        self._insert([(offset, doc_id)], commit=commit)

    def set_at_offset(self, offset: int, doc_id: str, commit: bool = True):
        offset = len(self) + offset if offset < 0 else offset
        sql = f'UPDATE {self.name} SET doc_id=? WHERE offset = ?'
        self._conn.execute(
            sql,
            (
                doc_id,
                offset,
            ),
        )
        if commit:
            self.commit()

    def shift_offset(
        self,
        shift_from: int,
        shift_step: int = 1,
        direction: str = 'left',
        commit: bool = True,
    ):
        if direction == 'left':
            sql = f'UPDATE {self.name} SET offset=offset-{shift_step} WHERE offset > ?'
        elif direction == 'right':
            sql = f'UPDATE {self.name} SET offset=offset+{shift_step} WHERE offset > ?'
        else:
            raise ValueError(
                f'The shit_offset directory `{direction}` is not supported!'
            )

        self._conn.execute(sql, (shift_from,))
        if commit:
            self._conn.commit()
