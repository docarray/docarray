import itertools
from typing import (
    Optional,
    TYPE_CHECKING,
    Generator,
    Iterator,
    Dict,
    Union,
    Iterable,
    MutableSequence,
    Sequence,
)

from .. import Document
from ..helper import typename

if TYPE_CHECKING:
    from ..types import DocumentArraySourceType, DocumentArrayIndexType


class DocumentArray(MutableSequence[Document]):
    def __init__(
        self, docs: Optional['DocumentArraySourceType'] = None, copy: bool = False
    ):
        super().__init__()
        self._data = []
        if docs is None:
            return
        elif isinstance(
            docs, (DocumentArray, Sequence, Generator, Iterator, itertools.chain)
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
    def _id2offset(self) -> Dict[str, int]:
        """Return the `_id_to_index` map

        :return: a Python dict.
        """
        if not hasattr(self, '_id_to_index'):
            self._rebuild_id2offset()
        return self._id_to_index

    def _rebuild_id2offset(self) -> None:
        """Update the id_to_index map by enumerating all Documents in self._data.

        Very costy! Only use this function when self._data is dramtically changed.
        """

        self._id_to_index = {
            d.id: i for i, d in enumerate(self._data)
        }  # type: Dict[str, int]

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        self._data.insert(index, value)
        self._id2offset[value.id] = index

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

    def __contains__(self, x: Union[str, 'Document']):
        if isinstance(x, str):
            return x in self._id2offset
        elif isinstance(x, Document):
            return x.id in self._id2offset
        else:
            return False

    def __getitem__(
        self, index: 'DocumentArrayIndexType'
    ) -> Union['Document', 'DocumentArray']:
        if isinstance(index, int):
            return self._data[index]
        elif isinstance(index, str):
            return self._data[self._id2offset[index]]
        elif isinstance(index, slice):
            return DocumentArray(self._data[index])
        elif index == Ellipsis:
            return self
        elif isinstance(index, Sequence):
            if isinstance(index[0], bool):
                return DocumentArray(itertools.compress(self._data, index))
            elif isinstance(index[0], int):
                return DocumentArray(self._data[t] for t in index)
            elif isinstance(index[0], str):
                return DocumentArray(self._data[self._id2offset[t]] for t in index)

        raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def __setitem__(
        self,
        index: 'DocumentArrayIndexType',
        value: Union['Document', Sequence['Document']],
    ):
        if isinstance(index, int):
            self._data[index] = value
            self._id2offset[value.id] = index
        elif isinstance(index, str):
            old_idx = self._id2offset.pop(index)
            self._data[old_idx] = value
            self._id2offset[value.id] = old_idx
        elif isinstance(index, slice):
            self._data[index] = value
            self._rebuild_id2offset()
        elif index == Ellipsis:
            self._data[index] = list(value)
            self._rebuild_id2offset()
        elif isinstance(index, Sequence):
            if isinstance(index[0], bool):
                if len(index) != len(self._data):
                    raise IndexError(
                        f'Boolean mask index is required to have the same length as {len(self._data)}, '
                        f'but receiving {len(index)}'
                    )
                _selected = itertools.compress(self._data, index)
                for _idx, _val in zip(_selected, value):
                    self[_idx.id] = _val
            elif isinstance(index[0], (int, str)):
                if not isinstance(value, Sequence) or len(index) != len(value):
                    raise ValueError(
                        f'Number of elements for assigning must be '
                        f'the same as the index length: {len(index)}'
                    )
                if isinstance(value, Document):
                    for si in index:
                        self[si] = value
                else:
                    for si, _val in zip(index, value):
                        self[si] = _val
        else:
            raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def __delitem__(self, index: 'DocumentArrayIndexType'):
        if isinstance(index, int):
            self._id2offset.pop(self._data[index].id)
            del self._data[index]
        elif isinstance(index, str):
            del self._data[self._id2offset[index]]
            self._id2offset.pop(index)
        elif isinstance(index, slice):
            del self._data[index]
            self._rebuild_id2offset()
        elif index == Ellipsis:
            self._data.clear()
            self._id2offset.clear()
        elif isinstance(index, Sequence):
            if isinstance(index[0], bool):
                self._data = list(
                    itertools.compress(self._data, (not _i for _i in index))
                )
                self._rebuild_id2offset()
            elif isinstance(index[0], int):
                for t in sorted(index, reverse=True):
                    del self[t]
            elif isinstance(index[0], str):
                for t in index:
                    del self[t]
        else:
            raise IndexError(f'Unsupported index type {typename(index)}: {index}')

    def append(self, value: 'Document'):
        """
        Append `doc` in :class:`DocumentArray`.

        :param value: The doc needs to be appended.
        """
        self._id2offset[value.id] = len(self._data)
        self._data.append(value)

    def extend(self, values: Iterable['Document']):
        """
        Extend the :class:`DocumentArray` by appending all the items from the iterable.

        :param values: the iterable of Documents to extend this array with
        """
        for doc in values:
            self.append(doc)

    def clear(self):
        """Clear the data of :class:`DocumentArray`"""
        self._data.clear()
        self._id2offset.clear()
