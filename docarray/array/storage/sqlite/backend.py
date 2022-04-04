import sqlite3
import warnings
from dataclasses import dataclass, field
from tempfile import NamedTemporaryFile
from typing import (
    Iterable,
    Dict,
    Optional,
    TYPE_CHECKING,
    Union,
)

from .helper import initialize_table
from ..base.backend import BaseBackendMixin
from ....helper import random_identity, dataclass_from_dict

if TYPE_CHECKING:
    from ....typing import DocumentArraySourceType


def _sanitize_table_name(table_name: str) -> str:
    ret = ''.join(c for c in table_name if c.isalnum() or c == '_')
    if ret != table_name:
        warnings.warn(f'The table name is changed to {ret} due to illegal characters')
    return ret


@dataclass
class SqliteConfig:
    connection: Optional[Union[str, 'sqlite3.Connection']] = None
    table_name: Optional[str] = None
    serialize_config: Dict = field(default_factory=dict)
    conn_config: Dict = field(default_factory=dict)
    journal_mode: str = 'DELETE'
    synchronous: str = 'OFF'


class BackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    schema_version = '0'

    def _sql(self, *args, **kwargs) -> 'sqlite3.Cursor':
        return self._cursor.execute(*args, **kwargs)

    def _commit(self):
        self._connection.commit()

    @property
    def _cursor(self) -> 'sqlite3.Cursor':
        return self._connection.cursor()

    def _init_storage(
        self,
        _docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[Union[SqliteConfig, Dict]] = None,
        **kwargs,
    ):
        if not config:
            config = SqliteConfig()

        if isinstance(config, dict):
            config = dataclass_from_dict(SqliteConfig, config)

        from docarray import Document

        sqlite3.register_adapter(
            Document, lambda d: d.to_bytes(**config.serialize_config)
        )
        sqlite3.register_converter(
            'Document', lambda x: Document.from_bytes(x, **config.serialize_config)
        )

        _conn_kwargs = dict()
        _conn_kwargs.update(config.conn_config)
        if config.connection is None:
            config.connection = NamedTemporaryFile().name

        if isinstance(config.connection, str):
            self._connection = sqlite3.connect(
                config.connection,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
                **_conn_kwargs,
            )
        elif isinstance(config.connection, sqlite3.Connection):
            self._connection = config.connection
        else:
            raise TypeError(
                f'connection argument must be None or a string or a sqlite3.Connection, not `{type(config.connection)}`'
            )
        self._connection.execute(f'PRAGMA synchronous={config.synchronous}')
        self._connection.execute(f'PRAGMA journal_mode={config.journal_mode}')

        self._table_name = (
            _sanitize_table_name(self.__class__.__name__ + random_identity())
            if config.table_name is None
            else _sanitize_table_name(config.table_name)
        )
        self._persist = bool(config.table_name)
        config.table_name = self._table_name
        initialize_table(
            self._table_name, self.__class__.__name__, self.schema_version, self._cursor
        )
        self._connection.commit()
        self._config = config

        super()._init_storage()

        if _docs is None:
            return
        elif isinstance(_docs, Iterable):
            self.clear()
            self.extend(_docs)
        else:
            self.clear()
            if isinstance(_docs, Document):
                self.append(_docs)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['_connection']
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        _conn_kwargs = dict()
        _conn_kwargs.update(state['_config'].conn_config)
        self._connection = sqlite3.connect(
            state['_config'].connection,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
            **_conn_kwargs,
        )
