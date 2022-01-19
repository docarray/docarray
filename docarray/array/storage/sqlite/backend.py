import dataclasses
from dataclasses import dataclass
from typing import (
    Optional,
    TYPE_CHECKING,
    Union,
    Dict,
)

from ..base.backend import BaseBackendMixin

if TYPE_CHECKING:
    import sqlite3

    from ....types import (
        DocumentArraySourceType,
    )


@dataclass
class SqliteConfig:
    connection: Optional[Union[str, 'sqlite3.Connection']] = None
    table_name: Optional[str] = None
    serialize_config: Optional[Dict] = None


class SqliteBackendMixin(BaseBackendMixin):
    """Provide necessary functions to enable this storage backend."""

    @property
    def schema_version(self) -> str:
        return '0'

    def _sql(self, *arg, **kwargs) -> 'sqlite3.Cursor':
        return self.connection.cursor().execute(*arg, **kwargs)

    def _insert_doc_at_idx(self, doc, idx: Optional[int] = None):
        if idx is None:
            idx = len(self)
        self._sql(
            f'INSERT INTO {self.table_name} (doc_id, serialized_value, item_order) VALUES (?, ?, ?)',
            (doc.id, doc, idx),
        )

    def _shift_index_right_backward(self, start: int):
        idx = len(self) - 1
        while idx >= start:
            self._sql(
                f"UPDATE {self.table_name} SET item_order = ? WHERE item_order = ?",
                (idx + 1, idx),
            )
            idx -= 1

    def _init_storage(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[SqliteConfig] = None,
    ):
        super().__init__(**(dataclasses.asdict(config) if config else {}))
        if docs is not None:
            self.clear()
            self.extend(docs)
