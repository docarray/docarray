from abc import abstractmethod
from typing import Iterator, Iterable, MutableSequence

from .... import Document


class BaseSequenceLikeMixin(MutableSequence[Document]):
    """Implement sequence-like methods"""

    def insert(self, index: int, value: 'Document'):
        """Insert `doc` at `index`.

        :param index: Position of the insertion.
        :param value: The doc needs to be inserted.
        """
        self._set_doc_by_id(value.id, value)
        self._offset2ids.insert(index, value.id)

    def append(self, value: 'Document'):
        """Append `doc` to the end of the array.

        :param value: The doc needs to be appended.
        """
        self._set_doc_by_id(value.id, value)
        self._offset2ids.append(value.id)

    @abstractmethod
    def __eq__(self, other):
        ...

    def __len__(self):
        return len(self._offset2ids)

    def __iter__(self) -> Iterator['Document']:
        for _id in self._offset2ids:
            yield self._get_doc_by_id(_id)

    @abstractmethod
    def __contains__(self, other):
        ...

    def clear(self):
        """Clear the data of :class:`DocumentArray`"""
        self._del_all_docs()

    def __bool__(self):
        """To simulate ```l = []; if l: ...```

        :return: returns true if the length of the array is larger than 0
        """
        return len(self) > 0

    def extend(self, values: Iterable['Document']) -> None:
        for value in values:
            self.append(value)
