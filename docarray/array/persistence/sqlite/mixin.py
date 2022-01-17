from dataclasses import dataclass
import dataclasses
from typing import Optional, TYPE_CHECKING, Callable, Union, cast

from .base import SqliteCollectionBase, RebuildStrategy
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
    rebuild_strategy: RebuildStrategy = RebuildStrategy.CHECK_WITH_FIRST_ELEMENT


class SqliteMixin(SqliteCollectionBase):
    """Enable SQLite persistence backend for DocumentArray.

    .. note::
        This has to be put in the first position when use it for subclassing
        i.e. `class SqliteDA(SqliteMixin, DA)` not the other way around.

    """
    _driver_class = _DictDatabaseDriver

    def __init__(self, docs: Optional['DocumentArraySourceType'] = None, config: Optional[SqliteConfig] = None):
        super().__init__(**(dataclasses.asdict(config) if config else {}))
        if docs is not None:
            self.clear()
            self.update(docs)

    def clear(self) -> None:
        cur = self.connection.cursor()
        self._driver_class.delete_all_records(self.table_name, cur)
        self.connection.commit()

    @property
    def schema_version(self) -> str:
        return "0"

    def _rebuild_check_with_first_element(self) -> bool:
        cur = self.connection.cursor()
        cur.execute(f"SELECT doc_id FROM {self.table_name} ORDER BY item_order LIMIT 1")
        res = cur.fetchone()
        return res is None

    def _do_rebuild(self) -> None:
        cur = self.connection.cursor()
        last_order = -1
        while last_order is not None:
            cur.execute(
                f"SELECT item_order FROM {self.table_name} WHERE item_order > ? ORDER BY item_order LIMIT 1",
                (last_order,),
            )
            res = cur.fetchone()
            if res is None:
                break
            i = res[0]
            cur.execute(
                f"SELECT doc_id, serialized_value FROM {self.table_name} WHERE item_order=?",
                (i,),
            )
            doc_id, serialized_value = cur.fetchone()
            cur.execute(
                f"UPDATE {self.table_name} SET doc_id=?, serialized_value=? WHERE item_order=?",
                (
                    doc_id,
                    serialized_value,
                    i,
                ),
            )
            last_order = i

    def serialize_value(self, value: VT) -> bytes:
        return self.value_serializer(value)

    def deserialize_value(self, value: bytes) -> VT:
        return self.value_deserializer(value)

    def __delitem__(self, key: KT) -> None:
        doc_id = self.serialize_key(key)
        cur = self.connection.cursor()
        if not self._driver_class.is_doc_id_in(self.table_name, cur, doc_id):
            raise KeyError(key)
        self._driver_class.delete_single_record_by_doc_id(self.table_name, cur, doc_id)
        self.connection.commit()

    def __getitem__(self, key: KT) -> VT:
        doc_id = self.serialize_key(key)
        cur = self.connection.cursor()
        serialized_value = self._driver_class.get_serialized_value_by_doc_id(
            self.table_name, cur, doc_id
        )
        if serialized_value is None:
            raise KeyError(key)
        return self.deserialize_value(serialized_value)

    def __iter__(self) -> Iterator[KT]:
        cur = self.connection.cursor()
        for doc_id in self._driver_class.get_doc_ids(self.table_name, cur):
            yield self.deserialize_key(doc_id)

    def __len__(self) -> int:
        cur = self.connection.cursor()
        return self._driver_class.get_count(self.table_name, cur)

    def __setitem__(self, key: KT, value: VT) -> None:
        doc_id = self.serialize_key(key)
        cur = self.connection.cursor()
        serialized_value = self.serialize_value(value)
        self._driver_class.upsert(self.table_name, cur, doc_id, serialized_value)
        self.connection.commit()

    def _create_volatile_copy(
            self,
            data: Optional[Mapping[KT, VT]] = None,
    ) -> "Dict[KT, VT]":

        return Dict[KT, VT](
            connection=self.connection,
            key_serializer=self.key_serializer,
            key_deserializer=self.key_deserializer,
            value_serializer=self.value_serializer,
            value_deserializer=self.value_deserializer,
            rebuild_strategy=RebuildStrategy.SKIP,
            persist=False,
            data=(self if data is None else data),
        )

    def copy(self) -> "Dict[KT, VT]":
        return self._create_volatile_copy()

    @classmethod
    def fromkeys(cls, iterable: Iterable[KT], value: Optional[VT]) -> "Dict[KT, VT]":
        raise NotImplementedError

    @overload
    def pop(self, k: KT) -> VT:
        ...

    @overload
    def pop(self, k: KT, default: Union[VT, T] = ...) -> Union[VT, T]:
        ...

    def pop(self, k: KT, default: Optional[Union[VT, object]] = None) -> Union[VT, object]:
        cur = self.connection.cursor()
        doc_id = self.serialize_key(k)
        serialized_value = self._driver_class.get_serialized_value_by_doc_id(
            self.table_name, cur, doc_id
        )
        if serialized_value is None:
            if default is None:
                raise KeyError(k)
            return default
        self._driver_class.delete_single_record_by_doc_id(self.table_name, cur, doc_id)
        self.connection.commit()
        return self.deserialize_value(serialized_value)

    def popitem(self) -> Tuple[KT, VT]:
        cur = self.connection.cursor()
        serialized_item = self._driver_class.get_last_serialized_item(self.table_name, cur)
        if serialized_item is None:
            raise KeyError("popitem(): dictionary is empty")
        self._driver_class.delete_single_record_by_doc_id(self.table_name, cur, serialized_item[0])
        self.connection.commit()
        return (
            self.deserialize_key(serialized_item[0]),
            self.deserialize_value(serialized_item[1]),
        )

    @overload
    def update(self, __other: Mapping[KT, VT], **kwargs: VT) -> None:
        ...

    @overload
    def update(self, __other: Iterable[Tuple[KT, VT]], **kwargs: VT) -> None:
        ...

    @overload
    def update(self, **kwargs: VT) -> None:
        ...

    def update(self, __other: Optional[Union[Iterable[Tuple[KT, VT]], Mapping[KT, VT]]] = None, **kwargs: VT) -> None:
        cur = self.connection.cursor()
        for k, v in chain(
                tuple() if __other is None else __other.items() if isinstance(__other, Mapping) else __other,
                cast(Mapping[KT, VT], kwargs).items(),
        ):
            self._driver_class.upsert(self.table_name, cur, self.serialize_key(k), self.serialize_value(v))
        self.connection.commit()

    def clear(self) -> None:
        cur = self.connection.cursor()
        self._driver_class.delete_all_records(self.table_name, cur)
        self.connection.commit()

    def __contains__(self, o: object) -> bool:
        return self._driver_class.is_doc_id_in(
            self.table_name, self.connection.cursor(), self.serialize_key(cast(KT, o))
        )

    @overload
    def get(self, key: KT) -> Union[VT, None]:
        ...

    @overload
    def get(self, key: KT, default_value: Union[VT, T]) -> Union[VT, T]:
        ...

    def get(self, key: KT, default_value: Optional[Union[VT, object]] = None) -> Union[VT, None, object]:
        doc_id = self.serialize_key(key)
        cur = self.connection.cursor()
        serialized_value = self._driver_class.get_serialized_value_by_doc_id(
            self.table_name, cur, doc_id
        )
        if serialized_value is None:
            return default_value
        return self.deserialize_value(serialized_value)

    def setdefault(self, key: KT, default: VT = None) -> VT:  # type: ignore
        doc_id = self.serialize_key(key)
        cur = self.connection.cursor()
        serialized_value = self._driver_class.get_serialized_value_by_doc_id(
            self.table_name, cur, doc_id
        )
        if serialized_value is None:
            self._driver_class.insert_serialized_value_by_doc_id(
                self.table_name, cur, doc_id, self.serialize_value(default)
            )
            return default
        return self.deserialize_value(serialized_value)

    @property
    def schema_version(self) -> str:
        return '0'