from typing import MutableSequence, TypeVar

from docarray.array.array.sequence_indexing_mixin import IndexingSequenceMixin

T_item = TypeVar('T_item')


class ListAdvanceIndex(IndexingSequenceMixin[T_item]):
    """
    A list wrapper that implement custom indexing

    You can index into a ListAdvanceIndex like a numpy array or torch tensor:

    .. code-block:: python
        da[0]  # index by position
        da[0:5:2]  # index by slice
        da[[0, 2, 3]]  # index by list of indices
        da[True, False, True, True, ...]  # index by boolean mask

    You can delete items from a DocumentArray like a Python List

    .. code-block:: python
        del da[0]  # remove first element from DocumentArray
        del da[0:5]  # remove elements for 0 to 5 from DocumentArray

    """

    _data: MutableSequence[T_item]

    def __init__(self, data: MutableSequence[T_item]):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self):
        return self._data
