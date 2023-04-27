from typing import TypeVar

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

    ...
