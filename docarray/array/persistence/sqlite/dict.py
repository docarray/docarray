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
            'serialized_key BLOB NOT NULL UNIQUE, '
            'serialized_value BLOB NOT NULL, '
            'item_order INTEGER PRIMARY KEY)'
        )

    @classmethod
    def delete_single_record_by_serialized_key(
        cls, table_name: str, cur: 'sqlite3.Cursor', serialized_key: bytes
    ) -> None:
        cur.execute(
            f'DELETE FROM {table_name} WHERE serialized_key=?', (serialized_key,)
        )

    @classmethod
    def delete_all_records(cls, table_name: str, cur: 'sqlite3.Cursor') -> None:
        cur.execute(f'DELETE FROM {table_name}')

    @classmethod
    def is_serialized_key_in(
        cls, table_name: str, cur: 'sqlite3.Cursor', serialized_key: bytes
    ) -> bool:
        cur.execute(
            f'SELECT 1 FROM {table_name} WHERE serialized_key=?', (serialized_key,)
        )
        return len(list(cur)) > 0

    @classmethod
    def get_serialized_value_by_serialized_key(
        cls, table_name: str, cur: 'sqlite3.Cursor', serialized_key: bytes
    ) -> Union[None, bytes]:
        cur.execute(
            f'SELECT serialized_value FROM {table_name} WHERE serialized_key=?',
            (serialized_key,),
        )
        res = cur.fetchone()
        if res is None:
            return None
        return cast(bytes, res[0])

    @classmethod
    def get_next_order(cls, table_name: str, cur: 'sqlite3.Cursor') -> int:
        cur.execute(f'SELECT MAX(item_order) FROM {table_name}')
        res = cur.fetchone()[0]
        if res is None:
            return 0
        return cast(int, res) + 1

    @classmethod
    def get_count(cls, table_name: str, cur: 'sqlite3.Cursor') -> int:
        cur.execute(f'SELECT COUNT(*) FROM {table_name}')
        res = cur.fetchone()
        return cast(int, res[0])

    @classmethod
    def get_serialized_keys(
        cls, table_name: str, cur: 'sqlite3.Cursor'
    ) -> Iterable[bytes]:
        cur.execute(f'SELECT serialized_key FROM {table_name} ORDER BY item_order')
        for res in cur:
            yield cast(bytes, res[0])

    @classmethod
    def insert_serialized_value_by_serialized_key(
        cls,
        table_name: str,
        cur: 'sqlite3.Cursor',
        serialized_key: bytes,
        serialized_value: bytes,
    ) -> None:
        item_order = cls.get_next_order(table_name, cur)
        cur.execute(
            f'INSERT INTO {table_name} (serialized_key, serialized_value, item_order) VALUES (?, ?, ?)',
            (serialized_key, serialized_value, item_order),
        )

    @classmethod
    def update_serialized_value_by_serialized_key(
        cls,
        table_name: str,
        cur: 'sqlite3.Cursor',
        serialized_key: bytes,
        serialized_value: bytes,
    ) -> None:
        cur.execute(
            f'UPDATE {table_name} SET serialized_value=? WHERE serialized_key=?',
            (serialized_value, serialized_key),
        )

    @classmethod
    def upsert(
        cls,
        table_name: str,
        cur: 'sqlite3.Cursor',
        serialized_key: bytes,
        serialized_value: bytes,
    ) -> None:
        if cls.is_serialized_key_in(table_name, cur, serialized_key):
            cls.update_serialized_value_by_serialized_key(
                table_name, cur, serialized_key, serialized_value
            )
        else:
            cls.insert_serialized_value_by_serialized_key(
                table_name, cur, serialized_key, serialized_value
            )

    @classmethod
    def get_last_serialized_item(
        cls, table_name: str, cur: 'sqlite3.Cursor'
    ) -> Tuple[bytes, bytes]:
        cur.execute(
            f'SELECT serialized_key, serialized_value FROM {table_name} ORDER BY item_order DESC LIMIT 1'
        )
        return cast(Tuple[bytes, bytes], cur.fetchone())

    @classmethod
    def get_reversed_serialized_keys(
        cls, table_name: str, cur: 'sqlite3.Cursor'
    ) -> Iterable[bytes]:
        cur.execute(f'SELECT serialized_key FROM {table_name} ORDER BY item_order DESC')
        for res in cur:
            yield cast(bytes, res[0])
