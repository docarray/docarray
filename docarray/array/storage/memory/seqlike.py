from typing import Iterator, Union, Sequence, Iterable, MutableSequence

from .... import Document


class SequenceLikeMixin(MutableSequence[Document]):
    """Implement sequence-like methods"""

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

    def clear(self):
        """Clear the data of :class:`DocumentArray`"""
        self._del_all_docs()

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def __repr__(self):
        return f'<DocumentArray (length={len(self)}) at {id(self)}>'

    def __add__(self, other: Union['Document', Sequence['Document']]):
        v = type(self)(self)
        v.extend(other)
        return v

    def extend(self, values: Iterable['Document']) -> None:
        self._data.extend(values)
        self._rebuild_id2offset()
