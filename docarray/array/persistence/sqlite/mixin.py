from dataclasses import dataclass
import dataclasses
from typing import Optional, TYPE_CHECKING, Callable, Union, cast, Iterable

from .base import SqliteCollectionBase
from .dict import _DictDatabaseDriver

if TYPE_CHECKING:
    import sqlite3

    from ....types import (
        T,
        Document,
        DocumentArraySourceType,
        DocumentArrayIndexType,
        DocumentArraySingletonIndexType,
        DocumentArrayMultipleIndexType,
        DocumentArrayMultipleAttributeType,
        DocumentArraySingleAttributeType,
    )


@dataclass
class SqliteConfig:
    connection: Optional[Union[str, 'sqlite3.Connection']] = None
    table_name: Optional[str] = None
    serializer: Optional[Callable[['Document'], bytes]] = None
    deserializer: Optional[Callable[[bytes], 'Document']] = None
    persist: bool = True


class SqliteMixin(SqliteCollectionBase):
    """Enable SQLite persistence backend for DocumentArray.

    .. note::
        This has to be put in the first position when use it for subclassing
        i.e. `class SqliteDA(SqliteMixin, DA)` not the other way around.

    """

    _driver_class = _DictDatabaseDriver

    def __init__(
        self,
        docs: Optional['DocumentArraySourceType'] = None,
        config: Optional[SqliteConfig] = None,
    ):
        super().__init__(**(dataclasses.asdict(config) if config else {}))
        if docs is not None:
            self.clear()
            self.extend(docs)

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        length = self._driver_class.get_max_index_plus_one(self.table_name, self.cursor)
        if index < 0:
            index = length + index
        index = max(0, min(length, index))
        self._driver_class.increment_indices(self.table_name, self.cursor, index)
        self._driver_class.insert_serialized_value_by_doc_id(
            self.table_name, self.cursor, value.id, self.serialize(value), index
        )
        self.connection.commit()

    def extend(self, values: Iterable['Document']) -> None:

        idx = self._driver_class.get_max_index_plus_one(self.table_name, self.cursor)
        for v in values:
            self._driver_class.insert_serialized_value_by_doc_id(
                self.table_name,
                cur=self.cursor,
                doc_id=v.id,
                serialized_value=self.serialize(v),
                item_order=idx,
            )
            idx += 1
        self.connection.commit()

    def clear(self) -> None:
        self._driver_class.delete_all_records(self.table_name, self.cursor)
        self.connection.commit()

    def __len__(self) -> int:
        return self._driver_class.get_count(self.table_name, self.cursor)

    @property
    def schema_version(self) -> str:
        return '0'
