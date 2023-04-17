from typing import Iterator, MutableSequence, TypeVar

from docarray.array.doc_list.sequence_indexing_mixin import IndexingSequenceMixin

T_item = TypeVar('T_item')


class ListAdvancedIndexing(IndexingSequenceMixin[T_item]):
    """
    A list wrapper that implements custom indexing

    You can index into a ListAdvanceIndex like a numpy array or torch tensor:

    ---

    ```python
    docs[0]  # index by position
    docs[0:5:2]  # index by slice
    docs[[0, 2, 3]]  # index by list of indices
    docs[True, False, True, True, ...]  # index by boolean mask
    ```

    ---

    """

    _data: MutableSequence[T_item]

    def __init__(self, data: MutableSequence[T_item]):
        self._data = data

    @property
    def data(self) -> MutableSequence[T_item]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T_item]:
        for item in self._data:
            yield item
