import sqlite3
import warnings
from abc import ABCMeta, abstractmethod
from tempfile import NamedTemporaryFile
from typing import Callable, Generic, Optional, TypeVar, Union, Dict
from uuid import uuid4

T = TypeVar('T')
KT = TypeVar('KT')
VT = TypeVar('VT')
_T = TypeVar('_T')
_S = TypeVar('_S')


def sanitize_table_name(table_name: str) -> str:
    ret = ''.join(c for c in table_name if c.isalnum() or c == '_')
    if ret != table_name:
        warnings.warn(f'The table name is changed to {ret} due to illegal characters')
    return ret


def create_random_name(suffix: str) -> str:
    return f"{suffix}_{str(uuid4()).replace('-', '')}"


class _SqliteCollectionBaseDatabaseDriver(metaclass=ABCMeta):
    @classmethod
    def initialize_metadata_table(cls, cur: sqlite3.Cursor) -> None:
        if not cls.is_metadata_table_initialized(cur):
            cls.do_initialize_metadata_table(cur)

    @classmethod
    def is_metadata_table_initialized(cls, cur: sqlite3.Cursor) -> bool:
        try:
            cur.execute('SELECT 1 FROM metadata LIMIT 1')
            _ = list(cur)
            return True
        except sqlite3.OperationalError as _:
            pass
        return False

    @classmethod
    def do_initialize_metadata_table(cls, cur: sqlite3.Cursor) -> None:
        cur.execute(
            '''
            CREATE TABLE metadata (
                table_name TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                container_type TEXT NOT NULL,
                UNIQUE (table_name, container_type)
            )
            '''
        )

    @classmethod
    def initialize_table(
        cls,
        table_name: str,
        container_type_name: str,
        schema_version: str,
        cur: sqlite3.Cursor,
    ) -> None:
        if not cls.is_table_initialized(
            table_name, container_type_name, schema_version, cur
        ):
            cls.do_create_table(table_name, container_type_name, schema_version, cur)
            cls.do_tidy_table_metadata(
                table_name, container_type_name, schema_version, cur
            )

    @classmethod
    def is_table_initialized(
        self,
        table_name: str,
        container_type_name: str,
        schema_version: str,
        cur: sqlite3.Cursor,
    ) -> bool:
        try:
            cur.execute(
                'SELECT schema_version FROM metadata WHERE table_name=? AND container_type=?',
                (table_name, container_type_name),
            )
            buf = cur.fetchone()
            if buf is None:
                return False
            version = buf[0]
            if version != schema_version:
                return False
            cur.execute(f'SELECT 1 FROM {table_name} LIMIT 1')
            _ = list(cur)
            return True
        except sqlite3.OperationalError as _:
            pass
        return False

    @classmethod
    def do_tidy_table_metadata(
        cls,
        table_name: str,
        container_type_name: str,
        schema_version: str,
        cur: sqlite3.Cursor,
    ) -> None:
        cur.execute(
            'INSERT INTO metadata (table_name, schema_version, container_type) VALUES (?, ?, ?)',
            (table_name, schema_version, container_type_name),
        )

    @classmethod
    @abstractmethod
    def do_create_table(
        cls,
        table_name: str,
        container_type_name: str,
        schema_version: str,
        cur: sqlite3.Cursor,
    ) -> None:
        ...

    @classmethod
    def drop_table(
        cls, table_name: str, container_type_name: str, cur: sqlite3.Cursor
    ) -> None:
        cur.execute(
            'DELETE FROM metadata WHERE table_name=? AND container_type=?',
            (table_name, container_type_name),
        )
        cur.execute(f'DROP TABLE {table_name}')

    @classmethod
    def alter_table_name(
        cls, table_name: str, new_table_name: str, cur: sqlite3.Cursor
    ) -> None:
        cur.execute(
            'UPDATE metadata SET table_name=? WHERE table_name=?',
            (new_table_name, table_name),
        )
        cur.execute(f'ALTER TABLE {table_name} RENAME TO {new_table_name}')


class SqliteCollectionBase(Generic[T], metaclass=ABCMeta):
    _driver_class = _SqliteCollectionBaseDatabaseDriver

    def __init__(
        self,
        connection: Optional[Union[str, sqlite3.Connection]] = None,
        table_name: Optional[str] = None,
        serialize_config: Optional[Dict] = None,
    ):
        super(SqliteCollectionBase, self).__init__()
        self._serialize_config = serialize_config or {}
        self._persist = not table_name

        from docarray import Document

        sqlite3.register_adapter(
            Document, lambda d: d.to_bytes(**self._serialize_config)
        )
        sqlite3.register_converter(
            'Document', lambda x: Document.from_bytes(x, **self._serialize_config)
        )

        _conn_kwargs = dict(detect_types=sqlite3.PARSE_DECLTYPES)
        if connection is None:
            self._connection = sqlite3.connect(
                NamedTemporaryFile().name, **_conn_kwargs
            )
        elif isinstance(connection, str):
            self._connection = sqlite3.connect(connection, **_conn_kwargs)
        elif isinstance(connection, sqlite3.Connection):
            self._connection = connection
        else:
            raise TypeError(
                f'connection argument must be None or a string or a sqlite3.Connection, not `{type(connection)}`'
            )

        self._table_name = (
            sanitize_table_name(create_random_name(self.container_type_name))
            if table_name is None
            else sanitize_table_name(table_name)
        )
        cur = self.connection.cursor()
        self._driver_class.initialize_metadata_table(cur)
        self._driver_class.initialize_table(
            self.table_name, self.container_type_name, self.schema_version, cur
        )
        self.connection.commit()

    def __del__(self) -> None:
        if not self._persist:
            cur = self.connection.cursor()
            self._driver_class.drop_table(
                self.table_name, self.container_type_name, cur
            )
            self.connection.commit()

    @property
    def table_name(self) -> str:
        return self._table_name

    @table_name.setter
    def table_name(self, table_name: str) -> None:
        cur = self.connection.cursor()
        new_table_name = sanitize_table_name(table_name)
        try:
            self._driver_class.alter_table_name(self.table_name, new_table_name, cur)
            self._table_name = new_table_name
        except sqlite3.IntegrityError as ex:
            raise ValueError(table_name) from ex

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def cursor(self) -> sqlite3.Cursor:
        return self.connection.cursor()

    @property
    def container_type_name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def schema_version(self) -> str:
        ...
