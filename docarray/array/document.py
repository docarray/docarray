import itertools
from collections.abc import MutableSequence
from typing import (
    Optional,
    TYPE_CHECKING,
    Generator,
    Iterator,
    Dict,
    Union,
    Iterable,
    Sequence,
)

from .. import Document
from ..helper import typename

if TYPE_CHECKING:
    from ..types import DocumentArraySourceType


class DocumentArray(MutableSequence):
    def __init__(
        self, docs: Optional['DocumentArraySourceType'] = None, copy: bool = False
    ):
        super().__init__()
        self._data = []
        if docs is None:
            return
        elif isinstance(
            docs, (DocumentArray, list, tuple, Generator, Iterator, itertools.chain)
        ):
            if copy:
                self._data.extend(Document(d, copy=True) for d in docs)
            elif isinstance(docs, DocumentArray):
                self._data = docs._data
            else:
                self._data.extend(docs)
        else:
            if isinstance(docs, Document):
                if copy:
                    self._data.append(Document(docs, copy=True))
                else:
                    self._data.append(docs)

    @property
    def _index_map(self) -> Dict:
        """Return the `_id_to_index` map

        :return: a Python dict.
        """
        if not hasattr(self, '_id_to_index'):
            self._rebuild_index_map()
        return self._id_to_index

    def _rebuild_index_map(self) -> None:
        """Update the id_to_index map by enumerating all Documents in self._data.

        Very costy! Only use this function when self._data is dramtically changed.
        """

        self._id_to_index = {
            d.id: i for i, d in enumerate(self._data)
        }  # type: Dict[str, int]

    def insert(self, index: int, doc: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param doc: The doc needs to be inserted.
        """
        self._data.insert(index, doc)
        self._index_map[doc.id] = index

    def __setitem__(self, key, value: 'Document'):
        if isinstance(key, int):
            self[key].copy_from(value)
            self._index_map[value.id] = key
        elif isinstance(key, str):
            self[self._index_map[key]].copy_from(value)
        else:
            raise IndexError(f'do not support this index type {typename(key)}: {key}')

    def __delitem__(self, index: Union[int, str, slice]):
        if isinstance(index, int):
            del self._data[index]
        elif isinstance(index, str):
            del self[self._index_map[index]]
            self._index_map.pop(index)
        elif isinstance(index, slice):
            del self._data[index]
        else:
            raise IndexError(
                f'do not support this index type {typename(index)}: {index}'
            )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and type(self._data) is type(other._data)
            and self._data == other._data
        )

    def __len__(self):
        return len(self._data)

    def __iter__(self) -> Iterator['Document']:
        yield from self._data

    def __contains__(self, item: str):
        return item in self._index_map

    def __getitem__(self, index: Union[int, str, slice, Sequence[int]]):
        if isinstance(index, int):
            return self._data[index]
        elif isinstance(index, str):
            return self[self._index_map[index]]
        elif isinstance(index, slice):
            return DocumentArray(self._data[index])
        elif isinstance(index, (list, tuple)):
            return DocumentArray(self._data[t] for t in index)
        else:
            IndexError(f'do not support this index type {typename(index)}: {index}')

    def append(self, doc: 'Document'):
        """
        Append `doc` in :class:`DocumentArray`.

        :param doc: The doc needs to be appended.
        """
        self._index_map[doc.id] = len(self._data)
        self._data.append(doc)

    def extend(self, docs: Iterable['Document']):
        """
        Extend the :class:`DocumentArray` by appending all the items from the iterable.

        :param docs: the iterable of Documents to extend this array with
        """
        if not docs:
            return

        for doc in docs:
            self.append(doc)

    def clear(self):
        """Clear the data of :class:`DocumentArray`"""
        self._data.clear()
        self._index_map.clear()
