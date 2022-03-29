from typing import Union, Iterable, MutableSequence, Iterator

from ..memory.backend import needs_id2offset_rebuild

from ..base.seqlike import BaseSequenceLikeMixin
from .... import Document


class SequenceLikeMixin(BaseSequenceLikeMixin):
    """Implement sequence-like methods"""

    @needs_id2offset_rebuild
    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        self._data.insert(index, value)

    def append(self, value: 'Document'):
        """Append `doc` to the end of the array.

        :param value: The doc needs to be appended.
        """
        self._data.append(value)
        if not self._needs_id2offset_rebuild:
            self._id_to_index[value.id] = len(self) - 1

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

    def __repr__(self):
        return f'<DocumentArray (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Iterable['Document']]):
        v = type(self)(self)
        v.extend(other)
        return v

    def extend(self, values: Iterable['Document']) -> None:
        values = list(values)  # consume the iterator only once
        last_idx = len(self._id2offset)
        self._data.extend(values)
        self._id_to_index.update({d.id: i + last_idx for i, d in enumerate(values)})
