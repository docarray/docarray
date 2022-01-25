from typing import Iterator, Union, Iterable, MutableSequence

from .... import Document


class SequenceLikeMixin(MutableSequence[Document]):
    """Implement sequence-like methods"""

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

    def append(self, value: 'Document') -> None:
        self._insert_doc_at_idx(value)

    def extend(self, values: Iterable['Document']) -> None:
        idx = len(self)
        for v in values:
            self._insert_doc_at_idx(v, idx)
            idx += 1

    def clear(self) -> None:
        raise NotImplementedError

    def __contains__(self, item: Union[str, 'Document']):
        if isinstance(item, str):
            raise NotImplementedError
            return len(list(r)) > 0
        elif isinstance(item, Document):
            return item.id in self  # fall back to str check
        else:
            return False

    def __len__(self) -> int:
        return self._pqlite.stat['doc_num']

    def __iter__(self) -> Iterator['Document']:
        ...
