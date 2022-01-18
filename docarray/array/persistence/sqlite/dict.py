from typing import Tuple, Union, cast, Iterable, TYPE_CHECKING

from .base import _SqliteCollectionBaseDatabaseDriver

if TYPE_CHECKING:
    import sqlite3


class _DictDatabaseDriver(_SqliteCollectionBaseDatabaseDriver):
    @classmethod
    def do_create_table(
        cls,
        table_name: str,
        container_type_nam: str,
        schema_version: str,
        cur: 'sqlite3.Cursor',
    ) -> None:
        cur.execute(
            f'CREATE TABLE {table_name} ('
            'doc_id TEXT NOT NULL UNIQUE, '
            'serialized_value Document NOT NULL, '
            'item_order INTEGER PRIMARY KEY)'
        )

    @classmethod
    def delete_single_record_by_doc_id(
        cls, table_name: str, cur: 'sqlite3.Cursor', doc_id: str
    ) -> None:
        cur.execute(f'DELETE FROM {table_name} WHERE doc_id=?', (doc_id,))

    @classmethod
    def get_next_order(cls, table_name: str, cur: 'sqlite3.Cursor') -> int:
        cur.execute(f'SELECT MAX(item_order) FROM {table_name}')
        res = cur.fetchone()[0]
        if res is None:
            return 0
        return cast(int, res) + 1

    @classmethod
    def get_doc_ids(cls, table_name: str, cur: 'sqlite3.Cursor') -> Iterable[str]:
        cur.execute(f'SELECT doc_id FROM {table_name} ORDER BY item_order')
        for res in cur:
            yield res[0]

    @classmethod
    def update_serialized_value_by_doc_id(
        cls,
        table_name: str,
        cur: 'sqlite3.Cursor',
        doc_id: str,
        serialized_value: bytes,
    ) -> None:
        cur.execute(
            f'UPDATE {table_name} SET serialized_value=? WHERE doc_id=?',
            (serialized_value, doc_id),
        )

    @classmethod
    def upsert(
        cls,
        table_name: str,
        cur: 'sqlite3.Cursor',
        doc_id: str,
        serialized_value: bytes,
    ) -> None:
        if cls.is_doc_id_in(table_name, cur, doc_id):
            cls.update_serialized_value_by_doc_id(
                table_name, cur, doc_id, serialized_value
            )
        else:
            cls.insert_serialized_value_by_doc_id(
                table_name, cur, doc_id, serialized_value
            )

    @classmethod
    def get_last_serialized_item(
        cls, table_name: str, cur: 'sqlite3.Cursor'
    ) -> Tuple[str, bytes]:
        cur.execute(
            f'SELECT doc_id, serialized_value FROM {table_name} ORDER BY item_order DESC LIMIT 1'
        )
        return cast(Tuple[str, bytes], cur.fetchone())

    @classmethod
    def get_reversed_doc_ids(
        cls, table_name: str, cur: 'sqlite3.Cursor'
    ) -> Iterable[str]:
        cur.execute(f'SELECT doc_id FROM {table_name} ORDER BY item_order DESC')
        for res in cur:
            yield cast(str, res[0])
