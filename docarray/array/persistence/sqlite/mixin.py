import itertools
from dataclasses import dataclass
import dataclasses
from typing import (
    Optional,
    TYPE_CHECKING,
    Callable,
    Union,
    cast,
    Iterable,
    Dict,
    Iterator,
    Sequence,
)

from .base import SqliteCollectionBase
from .dict import _DictDatabaseDriver
from ....helper import typename
import numpy as np

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

    from docarray import DocumentArray


@dataclass
class SqliteConfig:
    connection: Optional[Union[str, 'sqlite3.Connection']] = None
    table_name: Optional[str] = None
    serialize_config: Optional[Dict] = None


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
        length = len(self)
        if index < 0:
            index = length + index
        index = max(0, min(length, index))
        self._shift_index_right_backward(index)
        self._insert_doc_at_idx(doc=value, idx=index)
        self.connection.commit()

    def append(self, value: 'Document') -> None:
        self._insert_doc_at_idx(value)
        self.connection.commit()

    def extend(self, values: Iterable['Document']) -> None:
        idx = len(self)
        for v in values:
            self._insert_doc_at_idx(v, idx)
            idx += 1
        self.connection.commit()

    def clear(self) -> None:
        self._sql(f'DELETE FROM {self.table_name}')
        self.connection.commit()

    def __contains__(self, item: Union[str, 'Document']):
        if isinstance(item, str):
            r = self._sql(f"SELECT 1 FROM {self.table_name} WHERE doc_id=?", (item,))
            return len(list(r)) > 0
        elif isinstance(item, Document):
            return item.id in self  # fall back to str check
        else:
            return False

    def __len__(self) -> int:
        r = self._sql(f'SELECT COUNT(*) FROM {self.table_name}')
        return r.fetchone()[0]

    def __iter__(self) -> Iterator['Document']:
        r = self._sql(
            f'SELECT serialized_value FROM {self.table_name} ORDER BY item_order'
        )
        for res in r:
            yield res[0]

    def __getitem__(
        self, index: 'DocumentArrayIndexType'
    ) -> Union['Document', 'DocumentArray']:
        if isinstance(index, (int, np.generic)) and not isinstance(index, bool):
            return self._get_doc_by_offset(int(index))
        elif isinstance(index, str):
            if index.startswith('@'):
                return self.traverse_flat(index[1:])
            else:
                return self._get_doc_by_id(index)
        elif isinstance(index, slice):
            from docarray import DocumentArray

            return DocumentArray(self[j] for j in range(len(self))[index])
        elif index is Ellipsis:
            return self.flatten()
        elif isinstance(index, Sequence):
            from docarray import DocumentArray

            if (
                isinstance(index, tuple)
                and len(index) == 2
                and isinstance(index[0], (slice, Sequence))
            ):
                if isinstance(index[0], str) and isinstance(index[1], str):
                    # ambiguity only comes from the second string
                    if index[1] in self:
                        return DocumentArray([self[index[0]], self[index[1]]])
                    else:
                        return getattr(self[index[0]], index[1])
                elif isinstance(index[0], (slice, Sequence)):
                    _docs = self[index[0]]
                    _attrs = index[1]
                    if isinstance(_attrs, str):
                        _attrs = (index[1],)
                    return _docs._get_attributes(*_attrs)
            elif isinstance(index[0], bool):
                return DocumentArray(itertools.compress(self, index))
            elif isinstance(index[0], (int, str)):
                return DocumentArray(self[t] for t in index)
        elif isinstance(index, np.ndarray):
            index = index.squeeze()
            if index.ndim == 1:
                return self[index.tolist()]
            else:
                raise IndexError(
                    f'When using np.ndarray as index, its `ndim` must =1. However, receiving ndim={index.ndim}'
                )
        raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def _get_doc_by_offset(self, index: int) -> 'Document':
        r = self._sql(
            f"SELECT serialized_value FROM {self.table_name} WHERE item_order = ?",
            (index + (len(self) if index < 0 else 0),),
        )
        res = r.fetchone()
        if res is None:
            raise IndexError('index out of range')
        return res[0]

    def _get_doc_by_id(self, id: str) -> 'Document':
        r = self._sql(
            f"SELECT serialized_value FROM {self.table_name} WHERE doc_id = ?", (id,)
        )
        res = r.fetchone()
        if res is None:
            raise KeyError(f'Can not find Document with id=`{id}`')
        return res[0]

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
