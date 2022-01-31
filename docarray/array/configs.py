from dataclasses import dataclass, field
from typing import Optional, Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3


@dataclass
class SqliteConfig:
    connection: Optional[Union[str, 'sqlite3.Connection']] = None
    table_name: Optional[str] = None
    serialize_config: Dict = field(default_factory=dict)
    conn_config: Dict = field(default_factory=dict)
    journal_mode: str = 'DELETE'
    synchronous: str = 'OFF'
